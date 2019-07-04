# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark comparing dynamic_rnn implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autograph.examples.sysml2019 import benchmark_base
import numpy as np
from pyctr.api import conversion
from pyctr.examples.pytorch import pytorch
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import variables
import tensorflow as tf
import torch


tf.enable_eager_execution()


BATCH_SIZE = 32
MAX_SEQ_LEN = 100
FEATURE_SIZE = 50
HIDDEN_SIZE = 256

# TODO(b/129431400): Remove this alias.
RNNCell = torch.nn.RNNCell


def torch_dynamic_rnn(rnn_cell, input_data, initial_state, sequence_lengths):
  """A torch version of dynamic_rnn."""

  # [batch, time, features] -> [time, batch, features]
  input_data = input_data.permute(1, 0, 2)
  # Dimensions
  state = initial_state
  if sequence_lengths is None:
    shape = input_data.shape()
    max_seq_len = shape[0]
  else:
    max_seq_len = torch.max(sequence_lengths)

  for i in torch.arange(max_seq_len, dtype=torch.int64):
    new_state = rnn_cell(input_data[i], state)
    state = torch.where(i < sequence_lengths, new_state, state)
  return state


def _create_torch_rnn_cell(batch_size, input_size=FEATURE_SIZE):
  rnn_cell = RNNCell(input_size, HIDDEN_SIZE)
  init = [batch_size, HIDDEN_SIZE]
  return rnn_cell, torch.tensor(np.zeros(init, dtype=np.float32))


def _get_torch_inputs(input_data, sequence_lengths):
  """Convert input data to PyTorch format."""

  input_data = torch.tensor(input_data)
  sequence_lengths = torch.unsqueeze(torch.tensor(sequence_lengths), -1)

  return input_data, sequence_lengths


class RNNBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for variants of dynamic_rnn."""

  def _generate_fake_rnn_inputs(self, batch_size=BATCH_SIZE,
                                max_seq_len=MAX_SEQ_LEN):
    np.random.seed(17)

    input_data = np.random.random([batch_size, max_seq_len,
                                   FEATURE_SIZE]).astype(np.float32)
    # Generate some varying sequence lengths but keep max(sequence_lengths)
    # a constant, for more reproducible benchmarks.
    sequence_lengths = np.concatenate(([max_seq_len],
                                       np.random.randint(
                                           max_seq_len // 2,
                                           max_seq_len,
                                           size=[batch_size - 1]))).astype(
                                               np.int64)

    for i, seq_len in enumerate(sequence_lengths):
      input_data[i, seq_len:, :] = 0

    return input_data, sequence_lengths

  def _torch_baseline(self, batch_size, max_seq_len):
    """Benchmark stand-alone pytorch implementation."""

    input_data, sequence_lengths = self._generate_fake_rnn_inputs(
        batch_size=batch_size, max_seq_len=max_seq_len)
    rnn_cell, initial_state = _create_torch_rnn_cell(batch_size)
    input_data, sequence_lengths = _get_torch_inputs(input_data,
                                                     sequence_lengths)

    def target():
      torch_dynamic_rnn(rnn_cell, input_data, initial_state, sequence_lengths)

    self.time_execution(
        ('Torch', batch_size, max_seq_len),
        target,
        iter_volume=batch_size,
        iter_unit='examples',
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
        })

  def _torch_to_tf(self, batch_size, max_seq_len):
    """Benchmark pytorch converted to tensorflow graph implementation."""

    tf_dynamic_rnn = conversion.convert(torch_dynamic_rnn, pytorch,
                                        [functions, variables, control_flow])
    tf_create_rnn_cell = conversion.convert(
        _create_torch_rnn_cell, pytorch, [functions, variables, control_flow])
    tf_get_init_data = conversion.convert(_get_torch_inputs, pytorch,
                                          [functions, variables, control_flow])

    with tf.Graph().as_default():
      input_data, sequence_lengths = self._generate_fake_rnn_inputs(
          batch_size=batch_size, max_seq_len=max_seq_len)
      cell, init_state = tf_create_rnn_cell(batch_size)
      input_data, sequence_lengths = tf_get_init_data(input_data,
                                                      sequence_lengths)
      rnn_output = tf_dynamic_rnn(cell, input_data, init_state,
                                  sequence_lengths)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def target():
          sess.run(rnn_output)

        self.time_execution(
            ('Torch2Flow', batch_size, max_seq_len),
            target,
            iter_volume=batch_size,
            iter_unit='examples',
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
            })

  def benchmark_all(self):
    for batch_size in (32, 64, 128):
      for max_seq_len in (64, 128):
        self._torch_baseline(batch_size, max_seq_len)
        self._torch_to_tf(batch_size, max_seq_len)


if __name__ == '__main__':
  tf.test.main()

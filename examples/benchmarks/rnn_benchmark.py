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

import numpy as np
from pyctr.api import conversion
from pyctr.examples.models import dynamic_rnn_minimal
from pyctr.examples.numpy import numpy_to_tf
from pyctr.examples.numpy import numpy_to_torch
from pyctr.examples.pytorch import pytorch_to_numpy
from pyctr.examples.pytorch import pytorch_to_tf
from pyctr.examples.sysml2019 import benchmark_base
from pyctr.examples.tf import tf as tf_
from pyctr.examples.tf import tf_to_numpy
from pyctr.examples.tf import tf_to_pytorch
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import variables
import tensorflow as tf

tf.enable_eager_execution()


FEATURE_SIZE = 50
HIDDEN_SIZE = 256


class RNNBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for variants of dynamic_rnn."""

  def _numpy_baseline(self, batch_size, max_seq_len, input_size, hidden_size):
    inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_numpy(
        batch_size, max_seq_len, input_size, hidden_size)

    def target():
      dynamic_rnn_minimal.numpy(inputs, seq_len, w, b, init_state)

    self.time_execution(
        ('NumPy', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _numpy_to_eager(self, batch_size, max_seq_len, input_size, hidden_size):
    inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_tf(
        batch_size, max_seq_len, input_size, hidden_size)

    eager_from_np = conversion.convert(dynamic_rnn_minimal.numpy, numpy_to_tf,
                                       [variables, functions, control_flow])

    def target():
      eager_from_np(inputs, seq_len, w, b, init_state).numpy()

    self.time_execution(
        ('NumPy_Eager', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _numpy_to_tf(self, batch_size, max_seq_len, input_size, hidden_size):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        tensors = dynamic_rnn_minimal.random_inputs_tf(batch_size, max_seq_len,
                                                       input_size, hidden_size)
        inputs, seq_len, w, b, init_state = tensors

        tf_from_np = conversion.convert(dynamic_rnn_minimal.numpy, numpy_to_tf,
                                        [variables, functions, control_flow])
        ops = tf_from_np(inputs, seq_len, w, b, init_state)

        def target():
          sess.run(ops)

        self.time_execution(
            ('NumPy_TF', batch_size, max_seq_len, input_size, hidden_size),
            target,
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'input_size': input_size,
                'hidden_size': hidden_size,
            })

  def _numpy_to_pytorch(self, batch_size, max_seq_len, input_size, hidden_size):
    tensors = dynamic_rnn_minimal.random_inputs_torch(batch_size, max_seq_len,
                                                      input_size, hidden_size)
    inputs, seq_len, w, b, init_state = tensors

    torch_from_np = conversion.convert(dynamic_rnn_minimal.numpy,
                                       numpy_to_torch,
                                       [variables, functions, control_flow])

    def target():
      torch_from_np(inputs, seq_len, w, b, init_state).numpy()

    self.time_execution(
        ('NumPy_Torch', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _pytorch_baseline(self, batch_size, max_seq_len, input_size, hidden_size):
    inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_torch(
        batch_size, max_seq_len, input_size, hidden_size)

    def target():
      dynamic_rnn_minimal.pytorch(inputs, seq_len, w, b, init_state)

    self.time_execution(
        ('PyTorch', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _pytorch_to_eager(self, batch_size, max_seq_len, input_size, hidden_size):
    inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_tf(
        batch_size, max_seq_len, input_size, hidden_size)

    eager_from_torch = conversion.convert(dynamic_rnn_minimal.pytorch,
                                          pytorch_to_tf,
                                          [variables, functions, control_flow])

    def target():
      eager_from_torch(inputs, seq_len, w, b, init_state).numpy()

    self.time_execution(
        ('PyTorch_Eager', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _pytorch_to_numpy(self, batch_size, max_seq_len, input_size, hidden_size):
    tensors = dynamic_rnn_minimal.random_inputs_numpy(batch_size, max_seq_len,
                                                      input_size, hidden_size)
    inputs, seq_len, w, b, init_state = tensors

    numpy_from_torch = conversion.convert(dynamic_rnn_minimal.pytorch,
                                          pytorch_to_numpy,
                                          [variables, functions, control_flow])

    def target():
      np.copy(numpy_from_torch(inputs, seq_len, w, b, init_state))

    self.time_execution(
        ('PyTorch_NumPy', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _pytorch_to_tf(self, batch_size, max_seq_len, input_size, hidden_size):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        tensors = dynamic_rnn_minimal.random_inputs_tf(batch_size, max_seq_len,
                                                       input_size, hidden_size)
        inputs, seq_len, w, b, init_state = tensors

        tf_from_torch = conversion.convert(dynamic_rnn_minimal.pytorch,
                                           pytorch_to_tf,
                                           [variables, functions, control_flow])
        ops = tf_from_torch(inputs, seq_len, w, b, init_state)

        def target():
          sess.run(ops)

        self.time_execution(
            ('PYTorch_TF', batch_size, max_seq_len, input_size, hidden_size),
            target,
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'input_size': input_size,
                'hidden_size': hidden_size,
            })

  def _eager_baseline(self, batch_size, max_seq_len, input_size, hidden_size):
    inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_tf(
        batch_size, max_seq_len, input_size, hidden_size)

    def target():
      dynamic_rnn_minimal.eager(inputs, seq_len, w, b, init_state).numpy()

    self.time_execution(
        ('Eager', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _eager_to_numpy(self, batch_size, max_seq_len, input_size, hidden_size):
    tensors = dynamic_rnn_minimal.random_inputs_numpy(batch_size, max_seq_len,
                                                      input_size, hidden_size)
    inputs, seq_len, w, b, init_state = tensors

    numpy_from_eager = conversion.convert(dynamic_rnn_minimal.eager,
                                          tf_to_numpy,
                                          [variables, functions, control_flow])

    def target():
      np.copy(numpy_from_eager(inputs, seq_len, w, b, init_state))

    self.time_execution(
        ('Eager_NumPy', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _eager_to_pytorch(self, batch_size, max_seq_len, input_size, hidden_size):
    tensors = dynamic_rnn_minimal.random_inputs_torch(batch_size, max_seq_len,
                                                      input_size, hidden_size)
    inputs, seq_len, w, b, init_state = tensors

    torch_from_eager = conversion.convert(dynamic_rnn_minimal.eager,
                                          tf_to_pytorch,
                                          [variables, functions, control_flow])

    def target():
      torch_from_eager(inputs, seq_len, w, b, init_state).numpy()

    self.time_execution(
        ('Eager_PyTorch', batch_size, max_seq_len, input_size, hidden_size),
        target,
        extras={
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'input_size': input_size,
            'hidden_size': hidden_size,
        })

  def _eager_to_tf(self, batch_size, max_seq_len, input_size, hidden_size):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        tensors = dynamic_rnn_minimal.random_inputs_tf(batch_size, max_seq_len,
                                                       input_size, hidden_size)
        inputs, seq_len, w, b, init_state = tensors

        tf_from_eager = conversion.convert(dynamic_rnn_minimal.eager, tf_,
                                           [variables, functions, control_flow])
        ops = tf_from_eager(inputs, seq_len, w, b, init_state)

        def target():
          sess.run(ops)

        self.time_execution(
            ('Eager_TF', batch_size, max_seq_len, input_size, hidden_size),
            target,
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'input_size': input_size,
                'hidden_size': hidden_size,
            })

  def _tf_baseline(self, batch_size, max_seq_len, input_size, hidden_size):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        tensors = dynamic_rnn_minimal.random_inputs_tf(batch_size, max_seq_len,
                                                       input_size, hidden_size)
        inputs, seq_len, w, b, init_state = tensors

        ops = dynamic_rnn_minimal.tf_(inputs, seq_len, w, b, init_state)

        def target():
          sess.run(ops)

        self.time_execution(
            ('TF', batch_size, max_seq_len, input_size, hidden_size),
            target,
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'input_size': input_size,
                'hidden_size': hidden_size,
            })

  def _autograph_baseline(self, batch_size, max_seq_len, input_size,
                          hidden_size):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        inputs, seq_len, w, b, init_state = dynamic_rnn_minimal.random_inputs_tf(
            batch_size, max_seq_len, input_size, hidden_size)

        converted_fn = tf.autograph.to_graph(
            dynamic_rnn_minimal.eager, experimental_optional_features=None)
        ops = converted_fn(inputs, seq_len, w, b, init_state)

        def target():
          sess.run(ops)

        self.time_execution(
            ('AutoGraph', batch_size, max_seq_len, input_size, hidden_size),
            target,
            extras={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'input_size': input_size,
                'hidden_size': hidden_size,
            })

  def benchmark_all(self):
    for batch_size in (32,):
      for max_seq_len in (64,):

        self._eager_baseline(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._eager_to_numpy(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._eager_to_pytorch(
            batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._eager_to_tf(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)

        # TODO(mdanatg): Check correctness. It's suspiciously slow.
        self._numpy_baseline(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._numpy_to_eager(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._numpy_to_pytorch(
            batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._numpy_to_tf(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)

        self._pytorch_baseline(
            batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._pytorch_to_eager(
            batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._pytorch_to_numpy(
            batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._pytorch_to_tf(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)

        self._tf_baseline(batch_size, max_seq_len, FEATURE_SIZE, HIDDEN_SIZE)
        self._autograph_baseline(batch_size, max_seq_len, FEATURE_SIZE, 
                                 HIDDEN_SIZE)


if __name__ == '__main__':
  tf.test.main()

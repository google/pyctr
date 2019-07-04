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
"""Basic dynamic RNN implementation implemented in various frontends."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import numpy as np
import tensorflow as tf
import torch

# TODO(mdanatg): Accumulate RNN outputs once list staging is supported.
# TODO(mdanatg): Use mutator methods once supported.
# TODO(mdanatg): Add a *.where to RNN models for added realism.


def random_inputs_numpy(batch_size, max_seq_len, input_size, hidden_size):
  inputs = np.random.normal(size=(batch_size, 2 * max_seq_len, input_size))
  seq_len = np.arange(max_seq_len)  # Not random for equal computation.
  w = np.random.normal(size=(input_size + hidden_size, hidden_size))
  b = np.random.normal(size=(hidden_size,))
  init_state = np.zeros((batch_size, hidden_size))
  return inputs, seq_len, w, b, init_state


def random_inputs_tf(batch_size, max_seq_len, input_size, hidden_size):
  np_inputs = random_inputs_numpy(batch_size, max_seq_len, input_size,
                                  hidden_size)
  return tuple(tf.constant(a) for a in np_inputs)


def random_inputs_torch(batch_size, max_seq_len, input_size, hidden_size):
  np_inputs = random_inputs_numpy(batch_size, max_seq_len, input_size,
                                  hidden_size)
  return tuple(torch.tensor(a) for a in np_inputs)


def random_inputs_jax(batch_size, max_seq_len, input_size, hidden_size):
  np_inputs = random_inputs_numpy(batch_size, max_seq_len, input_size,
                                  hidden_size)
  return tuple(jax.numpy.array(a) for a in np_inputs)


def numpy(inputs, seq_len, w, b, init_state):
  """Very basic RNN model, implemented in NumPy."""
  inputs_time_major = np.transpose(inputs, axes=(1, 0, 2))
  max_seq_len = np.amax(seq_len)
  state = init_state
  for i in range(max_seq_len):
    x = inputs_time_major[i]
    h = np.concatenate((x, state), axis=1)
    state = np.tanh(np.dot(h, w) + b)  # Basic RNN cell
  return state


def eager(inputs, seq_len, w, b, init_state):
  """Very basic RNN model, implemented in TF Eager."""
  inputs_time_major = tf.transpose(inputs, (1, 0, 2))
  max_seq_len = tf.reduce_max(seq_len)
  state = init_state
  for i in range(max_seq_len):
    x = inputs_time_major[i]
    h = tf.concat((x, state), 1)
    state = tf.tanh(tf.linalg.matmul(h, w) + b)  # Basic RNN cell
  return state


def pytorch(inputs, seq_len, w, b, init_state):
  """Very basic RNN model, implemented in PyTorch."""
  inputs_time_major = torch.transpose(inputs, 1, 0)
  max_seq_len = torch.max(seq_len)
  state = init_state
  for i in range(max_seq_len):
    x = inputs_time_major[i]
    h = torch.cat((x, state), 1)
    state = torch.tanh(torch.mm(h, w) + b)  # Basic RNN cell
  return state


def tf_(inputs, seq_len, w, b, init_state):
  """Very basic RNN model, implemented in TF Graph."""
  inputs_time_major = tf.transpose(inputs, (1, 0, 2))
  max_seq_len = tf.reduce_max(seq_len)
  state = init_state

  def loop_body(i, state):
    x = inputs_time_major[i]
    h = tf.concat((x, state), 1)
    state = tf.tanh(tf.linalg.matmul(h, w) + b)  # Basic RNN cell
    return i + 1, state

  _, state = tf.while_loop(
      lambda i, state: i < max_seq_len,
      loop_body,
      (tf.zeros_like(max_seq_len), init_state))
  return state

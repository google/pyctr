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
"""Contains overloads to convert PyTorch to equivalent TensorFlow code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.examples.tf import tf as tf_
from pyctr.overloads import py_defaults
from pyctr.overloads import staging
import tensorflow as tf
import torch

init = py_defaults.init
assign = py_defaults.assign
if_stmt = tf_.if_stmt
while_stmt = tf_.while_stmt
for_stmt = tf_.for_stmt


def _patch_tensor_class(tf_tensor_class):
  """Monkey patch a TF tensor to achieve API compatibility with torch.Tensor."""
  # https://pytorch.org/docs/stable/tensors.html
  # https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor

  def permute(self, *dimensions):
    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
    # https://www.tensorflow.org/api_docs/python/tf/transpose
    dims = []
    for i in dimensions:
      dims.append(i)
    return tf.transpose(self, perm=dims)
  tf_tensor_class.permute = permute

  def size(self):
    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.size
    # https://www.tensorflow.org/api_docs/python/tf/size
    return tf.size(self)
  tf_tensor_class.size = size


_patch_tensor_class(tf.Tensor)


def read(var):
  assert isinstance(var, py_defaults.Variable)
  if isinstance(var.val, torch.Tensor):
    return tf.constant(var.val.numpy())
  return py_defaults.read(var)


call = staging.RewritingCallOverload(py_defaults.call)


def _torch2flow_dtype(torch_dtype):
  """Converts a given torch dtype to its TensorFlow equivalent."""
  # https://pytorch.org/docs/stable/tensors.html
  # https://www.tensorflow.org/api_docs/python/tf/dtypes/DType

  dtype_dict = {
      torch.float32: tf.float32,
      torch.float: tf.float32,
      torch.float64: tf.float64,
      torch.double: tf.float64,
      torch.float16: tf.float16,
      torch.half: tf.float16,
      torch.uint8: tf.uint8,
      torch.int16: tf.int16,
      torch.short: tf.int16,
      torch.int32: tf.int32,
      torch.int: tf.int32,
      torch.int64: tf.int64,
      torch.long: tf.int64,
      None: None,
  }
  return dtype_dict[torch_dtype]


@call.replaces(torch.rand)
def rand(*sizes):
  # https://pytorch.org/docs/stable/torch.html#torch.rand
  # https://www.tensorflow.org/api_docs/python/tf/random/uniform
  return tf.random.uniform(sizes)


@call.replaces(torch.nn.functional.conv1d)
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):  # pylint:disable=redefined-builtin, unused-argument
  # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.conv1d
  # https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
  if padding == 0:
    padding = 'VALID'
  else:
    raise NotImplementedError('Padding: {}'.format(padding))
  return tf.transpose(
      tf.nn.conv1d(
          tf.transpose(input, (0, 2, 1)),
          tf.transpose(weight, (2, 1, 0)),
          stride,
          padding=padding),
      (0, 2, 1),
  )


@call.replaces(torch.nn.RNNCell)
def rnn_cell(input_size, hidden_size, bias=True, nonlinearity='tanh'):  # pylint:disable=unused-argument
  """Convert the torch RNNCell implementation to the TF implementation."""
  # https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell
  # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell

  # TODO(aqj) Look into subclassing tf.nn.rnn_cell.BasicRNNCell instead
  class BasicRNNCellWrapper(object):
    """Wrapper around TF's RNN cell class to match the API to torch's."""
    # This is a prototype not all methods have been investigated.

    def __init__(self):
      self.cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size,
                                              activation=nonlinearity)
      self.is_built = False

    def __call__(self, inputs, state):
      if not self.is_built:
        self.cell.build(inputs.get_shape())
        self.is_built = True
      # torch does not return the new output, ignore it
      _, new_state = self.cell(inputs, state)
      return new_state

  return BasicRNNCellWrapper()


@call.replaces(torch.tensor)
def tensor(data, dtype=None, device=None, requires_grad=False):  # pylint:disable=unused-argument
  # https://pytorch.org/docs/stable/torch.html#torch.tensor
  # https://www.tensorflow.org/api_docs/python/tf/constant
  return tf.constant(value=data,
                     dtype=_torch2flow_dtype(dtype),
                     shape=None,
                     name='Const',
                     verify_shape=False)


@call.replaces(torch.max)
def max_fun(input_data):
  # https://pytorch.org/docs/stable/torch.html#torch.max
  # https://www.tensorflow.org/api_docs/python/tf/math/reduce_max
  return tf.reduce_max(input_data)


@call.replaces(torch.where)
def where(condition, x, y):
  # https://pytorch.org/docs/stable/torch.html#torch.where
  # https://www.tensorflow.org/api_docs/python/tf/where
  return tf.where(condition, x, y)


@call.replaces(torch.arange)
def arange(start, end=None, step=1,
           out=None, dtype=None, layout=torch.strided,  # pylint:disable=unused-argument
           device=None, requires_grad=False):  # pylint:disable=unused-argument
  # https://pytorch.org/docs/stable/torch.html#torch.arange
  # https://www.tensorflow.org/api_docs/python/tf/range
  return tf.range(start=start,
                  limit=end,
                  delta=step,
                  dtype=_torch2flow_dtype(dtype))


@call.replaces(torch.unsqueeze)
def unsqueeze(input_data, dim, out=None):  # pylint:disable=unused-argument
  # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
  # TODO(aqj) this is a no-op for now as its not needed for the tf version
  return input_data


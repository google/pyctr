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
"""Contains overloads to convert NumPy to equivalent TensorFlow code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyctr.examples.tf import tf as tf_
from pyctr.overloads import py_defaults
from pyctr.overloads import staging
import tensorflow as tf


init = py_defaults.init
assign = py_defaults.assign
if_stmt = tf_.if_stmt
for_stmt = tf_.for_stmt
while_stmt = tf_.while_stmt


def read(var):
  assert isinstance(var, py_defaults.Variable)
  if isinstance(var.val, np.ndarray):
    return tf.constant(var.val)
  return py_defaults.read(var)


call = staging.RewritingCallOverload(py_defaults.call)


@call.replaces(range)
def range_(n):
  return tf.range(n)


@call.replaces(np.transpose)
def transpose(x, axes):
  return tf.transpose(x, perm=axes)


@call.replaces(np.amax)
def amax(x):
  return tf.reduce_max(x)


@call.replaces(np.concatenate)
def concatenate(inputs, axis):
  return tf.concat(inputs, axis)


@call.replaces(np.tanh)
def tanh(x):
  return tf.tanh(x)


@call.replaces(np.dot)
def dot(x, y):
  # TODO(mdanatg): Rewrite using tf.dot.
  return tf.linalg.matmul(x, y)

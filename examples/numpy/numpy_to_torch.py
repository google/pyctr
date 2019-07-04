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
"""Contains overloads to convert NumPy to equivalent PyTorch code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyctr.overloads import py_defaults
from pyctr.overloads import staging
import torch

init = py_defaults.init
assign = py_defaults.assign
if_stmt = py_defaults.if_stmt
for_stmt = py_defaults.for_stmt
while_stmt = py_defaults.while_stmt


def read(var):
  assert isinstance(var, py_defaults.Variable)
  if isinstance(var.val, np.ndarray):
    return torch.tensor(var.val)
  return py_defaults.read(var)


call = staging.RewritingCallOverload(py_defaults.call)


@call.replaces(np.transpose)
def transpose(x, axes):
  changed = tuple(a for i, a in enumerate(axes) if i != a)
  if len(changed) != 2:
    raise ValueError('PyTorch doesnt support swapping more than two dims?')
  dim0, dim1 = changed
  return torch.transpose(x, dim0, dim1)


@call.replaces(np.amax)
def amax(x):
  return torch.max(x)


@call.replaces(np.concatenate)
def concatenate(inputs, axis):
  return torch.cat(inputs, axis)


@call.replaces(np.tanh)
def tanh(x):
  return torch.tanh(x)


@call.replaces(np.dot)
def dot(x, y):
  return torch.mm(x, y)

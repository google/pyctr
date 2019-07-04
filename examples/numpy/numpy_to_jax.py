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
"""Contains overloads to convert NumPy to equivalent JAX code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import numpy as jnp
import numpy as np
from pyctr.examples.jax import jax
from pyctr.overloads import py_defaults
from pyctr.overloads import staging

init = py_defaults.init
assign = py_defaults.assign
if_stmt = jax.if_stmt
for_stmt = jax.for_stmt
while_stmt = jax.while_stmt


def read(var):
  assert isinstance(var, py_defaults.Variable)
  if isinstance(var.val, np.ndarray):
    return jnp.array(var.val)
  return py_defaults.read(var)


call = staging.RewritingCallOverload(py_defaults.call)


@call.replaces(np.transpose)
def transpose(x, axes):
  return jnp.transpose(x, axes)


call.replaces(np.amax)(jnp.amax)
call.replaces(np.concatenate)(jnp.concatenate)
call.replaces(np.tanh)(jnp.tanh)
call.replaces(np.dot)(jnp.dot)


@call.replaces(range)
def range_(n):
  return jnp.arange(n)

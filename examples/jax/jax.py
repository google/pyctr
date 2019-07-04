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
"""Contains overloads to convert Python to equivalent JAX code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import lax
from pyctr.overloads import py_defaults
from pyctr.overloads import staging


init = py_defaults.init
assign = py_defaults.assign
read = py_defaults.read
call = py_defaults.call


def if_stmt(cond, body, orelse, local_writes):
  """Functional form of an if statement.

  Args:
    cond: Callable with no arguments, predicate of conditional.
    body: Callable with no arguments, and outputs of the positive (if) branch as
      return type.
    orelse: Callable with no arguments, and outputs of the negative (else)
      branch as return type.
    local_writes: list(pyct.Variable), list of variables assigned in either body
      or orelse.

  Returns:
    Tuple containing the statement outputs.
  """

  cond_result = cond()

  def if_body(*_):
    modified_vals, _ = staging.execute_isolated(body, local_writes)
    return modified_vals

  def if_orelse(*_):
    modified_vals, _ = staging.execute_isolated(orelse, local_writes)
    return modified_vals

  result_values = lax.cond(cond_result, (), if_body, (), if_orelse)
  for var, retval in zip(local_writes, result_values):
    var.val = retval

  return result_values


def while_stmt(cond, body, _, local_writes):
  """Functional form of a while statement."""

  local_writes = [
      var for var in local_writes if not py_defaults.is_undefined(var.val)
  ]

  def while_test(state):
    for var, s in zip(local_writes, state):
      var.val = s
    _, result_values = staging.execute_isolated(cond, local_writes)
    return result_values

  def while_body(state):
    for var, s in zip(local_writes, state):
      var.val = s
    modified_vals, _ = staging.execute_isolated(body, local_writes)
    return modified_vals

  result_values = lax.while_loop(while_test, while_body,
                                 [var.val for var in local_writes])

  for var, val in zip(local_writes, result_values):
    var.val = val

  return result_values


def for_stmt(target, iter_, body, orelse, modified_vars):
  """Functional form of a for statement."""
  del orelse

  modified_vars = [
      var for var in modified_vars if not py_defaults.is_undefined(var.val)
  ]

  def for_body(idx, state):
    for var, s in zip(modified_vars, state):
      var.val = s
    target.val = iter_[idx]
    modified_vals, _ = staging.execute_isolated(body, modified_vars)
    return modified_vals

  results = lax.fori_loop(0, len(iter_), for_body,
                          [var.val for var in modified_vars])
  for var, val in zip(modified_vars, results):
    var.val = val

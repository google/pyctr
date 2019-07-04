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
"""Contains overloads to convert Python to equivalent z3py code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.overloads import py_defaults
from pyctr.overloads import staging
import z3


init = py_defaults.init
assign = py_defaults.assign
read = py_defaults.read


call = staging.RewritingCallOverload(py_defaults.call)


@call.replaces(abs)
def z3_abs(x):
  if isinstance(x, z3.ArithRef):
    return z3.If(x < 0, -x, x)
  else:
    return abs(x)


def and_(a, b):
  """Overload of `and` which builds z3.And statements.

  Eagerly simplifies the expression if any of the operands are Python booleans.
  Otherwise, a z3.And is generated for z3.BoolRef objects.

  Args:
    a: Union[bool, z3.BoolRef], first operand of `and`
    b: Tuple[Callable[[], Union[bool, z3.BoolRef]]], lazy thunks for remaining
        operands

  Returns:
    corresponding z3.And expression, or a Python expression if no z3.BoolRefs.
  """
  assert isinstance(b, tuple)
  if not b:
    return a
  if isinstance(a, z3.BoolRef):
    return_val = and_(b[0](), b[1:])
    if isinstance(return_val, z3.BoolRef):
      return z3.And(a, return_val)
    else:
      if return_val:
        return a
      else:
        return False
  else:
    if a:
      return and_(b[0](), b[1:])
    else:
      return False


def or_(a, b):
  """Overload of `or` which builds z3.Or statements.

  Eagerly simplifies the expression if any of the operands are Python booleans.
  Otherwise, a z3.Or is generated for z3.BoolRef objects.

  Args:
    a: Union[bool, z3.BoolRef], first operand of `or`
    b: Tuple[Callable[[], Union[bool, z3.BoolRef]]], lazy thunks for remaining
        operands

  Returns:
    corresponding z3.Or expression, or a Python expression if no z3.BoolRefs.
  """
  assert isinstance(b, tuple)
  if not b:
    return a
  if isinstance(a, z3.BoolRef):
    return_val = or_(b[0](), b[1:])

    if isinstance(return_val, z3.BoolRef):
      return z3.Or(a, return_val)
    else:
      if return_val:
        return True
      else:
        return a
  else:
    if a:
      return True
    else:
      return or_(b[0](), b[1:])


def not_(x):
  if isinstance(x, z3.BoolRef):
    return z3.Not(x)
  else:
    return not x


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

  if isinstance(cond_result, z3.BoolRef):
    body_vals, _ = staging.execute_isolated(body, local_writes)
    orelse_vals, _ = staging.execute_isolated(orelse, local_writes)

    for body_result, else_result, modified_var in zip(body_vals, orelse_vals,
                                                      local_writes):
      # Unlike e.g., TensorFlow, z3 does not do tracing on If statements.
      # Instead, it expects the results of the body and orelse branches passed
      # as values. As such, each result is the result of the deferred z3.If
      # statement.
      modified_var.val = z3.If(cond_result, body_result, else_result)
  else:
    py_defaults.if_stmt(lambda: cond_result, body, orelse, local_writes)

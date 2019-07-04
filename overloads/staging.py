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
"""Contains common methods useful for DSL developers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.overloads import py_defaults


# TODO(mdanatg): Tests.
class RewritingCallOverload(object):
  """A function call overload that can replace select functions.

  Objects of this class can serve as function call overloads. They will replace
  function calls that have been decorated with the object's `replaces`
  decorator.

  Example:

    call = RewritingCallOverload(py_defaults.call)

    # Calls to "foo" will be replaced by "bar".
    @call.replaces(foo)
    def bar():
      pass
  """

  def __init__(self, default_call):
    self._registry = {}
    self._default_call = default_call

  def replaces(self, original):
    """Decorator registering a function as replacement of `original`."""
    original_id = id(original)
    existing = self._registry.get(original_id, None)
    if existing is not None:
      raise ValueError('{} already replaced by {}'.format(original, existing))

    def wrapper(f):
      self._registry[original_id] = f
      return f

    return wrapper

  def __call__(self, f, args, kwargs):
    if id(f) in self._registry:
      return self._registry[id(f)](*args, **kwargs)
    return self._default_call(f, args, kwargs)


def run_python_while(cond, body, orelse, init_cond_result):
  if init_cond_result:
    body()
    return py_defaults.while_stmt(cond, body, orelse, None)
  else:
    return orelse()


# TODO(jmd1011): Pull out functionality regarding return values into separate
# function.
def execute_isolated(func, func_freevars):
  """Executes func and restores variables to original state.

  This is used to allow for tracing of control flow blocks (e.g., tf.cond).
  Staged conditionals (e.g., tf.cond) will execute both branches, so this
  restores the state after running each branch (while returning the correct
  values).

  Due to the virtualization of variables, we cannot simply create new symbols,
  as functions we call and which we have no control over must track updates.

  Note: func is expected to be impure, and may modify the variables in
  func_freevars.

  Args:
    func: Callable with no arguments, either body or orelse from if_stmt.
    func_freevars: all variables modified either by func or func's counterpart
      if applicable (e.g., if/else branches).

  Returns:
    modified_vals: the values of all variables in func_freevars after executing
      func.
  """

  original_vals = [var.val for var in func_freevars]
  return_vals = func()
  modified_vals = [var.val for var in func_freevars]
  for var, val in zip(func_freevars, original_vals):
    var.val = val
  return modified_vals, return_vals

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
"""Contains overloads to convert Python to equivalent TensorFlow code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.overloads import py_defaults
from pyctr.overloads import staging

import tensorflow as tf


init = py_defaults.init
assign = py_defaults.assign
read = py_defaults.read


call = staging.RewritingCallOverload(py_defaults.call)


@call.replaces(range)
def range_(r):
  if tf.is_tensor(r):
    return tf.range(r)
  else:
    return range(r)


def _filter_undefined(all_symbols):
  """Returns the names of undefined symbols contained in all_symbols."""
  undefined_symbols = [
      s.name for s in all_symbols if py_defaults.is_undefined(s)
  ]
  return undefined_symbols


def _wrap_in_protection_from_undefined(func, branch_name):
  """Wraps function to raise useful error when it returns undefined symbols."""

  def protected_func():
    """Calls function and raises an error if undefined symbols are returned."""
    results = func()
    undefined_symbols = None
    if isinstance(results, tuple):
      undefined_symbols = _filter_undefined(results)
    elif py_defaults.is_undefined(results):
      # Single return value
      undefined_symbols = results.symbol_name

    if undefined_symbols:
      message = ('The following symbols must also be initialized in the %s '
                 'branch: {}. Alternatively, you may initialize them before '
                 'the if statement.') % branch_name
      message = message.format(undefined_symbols)
      raise ValueError(message)
    return results

  return protected_func


def _tf_if_stmt(cond, body, orelse):
  """Overload of if_stmt that stages a TF cond."""
  protected_body = _wrap_in_protection_from_undefined(body, branch_name='if')
  protected_orelse = _wrap_in_protection_from_undefined(
      orelse, branch_name='else')

  return tf.cond(cond, protected_body, protected_orelse)


def if_stmt(cond, body, orelse, local_writes):
  """Functional form of an if statement.

  Args:
    cond: Boolean.
    body: Callable with no arguments, and outputs of the positive (if) branch as
      return type.
    orelse: Callable with no arguments, and outputs of the negative (else)
      branch as return type.
    local_writes: list(pyct.Variable), list of variables assigned in either body
      or orelse

  Returns:
    Tuple containing the statement outputs.
  """
  cond_result = cond()
  if tf.is_tensor(cond_result):
    def if_body(*_):
      modified_vals, _ = staging.execute_isolated(body, local_writes)
      return modified_vals

    def if_orelse(*_):
      modified_vals, _ = staging.execute_isolated(orelse, local_writes)
      return modified_vals

    result_values = _tf_if_stmt(cond_result, if_body, if_orelse)

    for var, val in zip(local_writes, result_values):
      var.val = val

    return result_values
  else:
    return py_defaults.if_stmt(lambda: cond_result, body, orelse,
                               local_writes)


def _tf_while_stmt(cond, body, local_writes):
  """Overload of while_stmt that stages a TF while_stmt."""

  # Non-v2 while_loop unpacks the results when there is only one return value.
  # This enforces consistency across versions.
  opts = {'return_same_structure': True}

  return tf.while_loop(cond, body, local_writes, **opts)


def while_stmt(cond, body, orelse, local_writes):
  """Functional form of a while statement."""

  cond_result = cond()

  if tf.is_tensor(cond_result):
    local_writes = [
        var for var in local_writes if not py_defaults.is_undefined(var.val)
    ]

    def while_test(*state):
      for var, s in zip(local_writes, state):
        var.val = s
      _, retvals = staging.execute_isolated(cond, local_writes)
      return retvals

    def while_body(*state):
      for var, s in zip(local_writes, state):
        var.val = s
      mods, _ = staging.execute_isolated(body, local_writes)
      return mods

    result_values = _tf_while_stmt(while_test, while_body,
                                   [var.val for var in local_writes])

    for var, val in zip(local_writes, result_values):
      var.val = val
  else:
    staging.run_python_while(cond, body, orelse, cond_result)


def _tf_len(s):
  """Overload of len_ for Tensor arguments."""
  # Statically shaped tensors: length is known ahead of time.
  if s.shape.ndims and s.shape.dims[0].value is not None:
    return s.shape.dims[0].value

  # Static shape of unknown dimensions: use dynamic shape but statically
  # chech that it's a scalar.
  shape = tf.shape(s)

  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)

  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))

  if shape.shape.dims[0].value is not None:
    return tf.shape(s)[0]

  # Fully dynamic shape: use ops.
  rank = tf.rank(s)

  def raise_zero_rank_error():
    msg = tf.string_join(
        ['len requires non-zero rank, got ',
         tf.as_string(rank)])
    with tf.control_dependencies([tf.Assert(False, [msg])]):
      return tf.constant(0, dtype=tf.dtypes.int32)

  return tf.cond(rank > 0, lambda: tf.shape(s)[0], raise_zero_rank_error)


def for_stmt(target, iter_, body, orelse, local_writes):
  """Functional form of a for statement."""

  if tf.is_tensor(iter_):
    local_writes = [
        var for var in local_writes if not py_defaults.is_undefined(var.val)
    ]

    n = _tf_len(iter_)

    def for_test(i, *_):
      return i < n

    def for_body(iterate_index, *state):  # pylint: disable=missing-docstring
      for var, s in zip(local_writes, state):
        var.val = s
      target.val = iter_[iterate_index]
      mods, _ = staging.execute_isolated(body, local_writes)

      state = [iterate_index + 1] + mods
      return state

    result_values = _tf_while_stmt(for_test, for_body,
                                   [0] + [var.val for var in local_writes])

    for var, val in zip(local_writes, result_values[1:]):
      var.val = val
  else:
    py_defaults.for_stmt(target, iter_, body, orelse, local_writes)

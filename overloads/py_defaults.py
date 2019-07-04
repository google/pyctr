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
"""Contains functions modeling default behavior for Python statements."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(jmd1011): Add remaining statement types.


def call(func, args, keywords):
  return func(*args, **keywords)


# Staged if statements will have a fourth argument: local_writes. This is to
# inventory variables written to in branches. These aren't necessary in the
# default Python semantics.
def if_stmt(cond, body, orelse, _):
  if cond():
    body()
  else:
    orelse()


# Staged while loops will have a fourth argument: local_writes. This is to
# inventory variables written to in the body of the loop. These aren't necessary
# in the default Python semantics.
# TODO(b/123998025): Handle break/continue
def while_stmt(cond, body, orelse, _):
  while cond():
    body()
  else:  # pylint: disable=useless-else-on-loop
    orelse()


# Staged for loops will have a fourth argument: local_writes. This is to
# inventory variables written to in the body of the loop. These aren't necessary
# in the default Python semantics.
# TODO(b/123998025): Handle break/continue
def for_stmt(target, iter_, body, orelse, _):
  for i in range(len(iter_)):
    target.val = iter_[i]
    body()
  else:  # pylint: disable=useless-else-on-loop
    orelse()


def and_(x, operands):
  if x:
    for op in operands:
      if not op():
        return False

    return True
  else:
    return False


def or_(x, operands):
  if not x:
    for op in operands:
      if op():
        return True

    return False
  else:
    return True


def not_(x):
  return not x


class Undefined(object):

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return 'pyct.Undefined({})'.format(self.name)


def is_undefined(value):
  """Checks whether Pyct has determined that a given value is undefined.

  This will return True if py_defaults.init has been called on a
  py_defaults.Variable, but py_defaults.assign has not.

  Args:
    value: value to test for undefinedness

  Returns:
    Boolean, whether the input value is undefined.
  """
  return isinstance(value, Undefined)


class Variable(object):

  def __init__(self, val, name):
    self.val = val
    self.name = name

  def __getitem__(self, key):
    return self.val[key]

  def __repr__(self):
    return 'pyct.Variable(name={}, val={})'.format(self.name, self.val)


def init(name):
  return Variable(Undefined(name), name)


def assign(lhs, rhs):
  lhs.val = rhs
  return lhs


class PyctUnboundLocalError(Exception):
  pass


def read(var):
  assert isinstance(var, Variable)
  if isinstance(var.val, Undefined):
    raise PyctUnboundLocalError(
        'local variable \'{}\' referenced before assignment'.format(
            var.val.name))
  elif isinstance(var.val, Variable):
    # This is to handle for loop targets which are only converted to Variable
    # when for loops are virtualized.
    # TODO(jmd1011): It may be more consistent to always convert, though this
    # creates an odd pattern in the generated code.
    return read(var.val)

  return var.val

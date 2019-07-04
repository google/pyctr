# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed thnd in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains functions modeling default behavior for Python statements."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.overloads import py_defaults


def _handler(x, name):
  if x is None:
    return None
  return getattr(type(x), name, None)


def if_stmt(cond, body, orelse, closure):
  hnd = _handler(cond, '__if__')
  if hnd is not None:
    hnd(cond, body, orelse, closure)
  else:
    py_defaults.if_stmt(cond, body, orelse, closure)


def while_stmt(cond, body, orelse, closure):
  hnd = _handler(cond, '__while__')
  if hnd is not None:
    hnd(cond, body, orelse, closure)
  else:
    py_defaults.while_stmt(cond, body, orelse, closure)


def not_(x):
  hnd = _handler(x, '__not__')
  if hnd is not None:
    return hnd(x)
  else:
    return py_defaults.not_(x)


def assign(lhs, rhs):
  hnd = _handler(lhs, '__assign__')
  if hnd is not None:
    return hnd(lhs, rhs)
  else:
    return py_defaults.assign(lhs, rhs)


def read(var):
  hnd = _handler(var, '__read__')
  if hnd is not None:
    return hnd(var)
  else:
    return py_defaults.read(var)

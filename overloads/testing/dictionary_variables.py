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


variables = {}
suffix = 0


class Undefined(object):

  def __init__(self, name):
    self.name = name


def fresh_name():
  global suffix
  name = 'x_{}'.format(suffix)
  suffix += 1
  return name


def init(name):
  var = fresh_name()
  variables[var] = Undefined(name)
  return var


def assign(lhs, rhs):
  variables[lhs] = rhs
  return lhs


def read(var):
  val = variables[var]

  if isinstance(val, Undefined):
    raise UnboundLocalError(
        'local variable \'{}\' referenced before assignment'.format(
            val.name))

  return val

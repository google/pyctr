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
"""Contains a test module for testing conversion.py with variable virtualization.
"""


class Undefined(object):

  def __init__(self, name):
    self.name = name


class Variable(object):

  def __init__(self, val):
    self.val = val


def init(name):
  return Variable(Undefined(name))


def assign(lhs, rhs):
  lhs.val = rhs
  return lhs


def read(var):
  if isinstance(var.val, int):
    # For testing purposes, return value + 1 for each read
    return var.val + 1
  elif isinstance(var.val, Undefined):
    raise UnboundLocalError(
        'local variable \'{}\' referenced before assignment'.format(
            var.val.name))
  return var.val

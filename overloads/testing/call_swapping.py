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
"""Contains functions which swap out function calls."""
from pyctr.overloads import py_defaults


def add(x, y):
  return x + y


def _add_one(x, y):
  if isinstance(x, int):
    return x + y + 1
  else:
    return x + y


function_map = {add: _add_one}


def call(func, args, keywords):
  if func in function_map:
    return function_map[func](*args, **keywords)
  else:
    return py_defaults.call(func, args, keywords)

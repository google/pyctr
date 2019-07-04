# python3

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
"""Tests for nonlocal variables using variables converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.transformers.virtualization import variables


class VariablesPy3Test(test.TestCase):

  def default_convert(self, func, *args):
    converted_func = conversion.convert(func, py_defaults, [variables])
    return converted_func(*args)

  def test_nonlocal_transform(self):

    def nonlocal_closing_local():
      x = 1

      def inner_func():
        nonlocal x
        x = 2

      inner_func()
      return x

    self.assertEqual(
        self.default_convert(nonlocal_closing_local), nonlocal_closing_local())

  def test_nonlocal_no_transform(self):

    def nonlocal_closing_free():
      x = 1

      def mod_x(y):
        nonlocal x
        x = y

      def read_x():
        return x

      return mod_x, read_x

    mod_x, read_x = nonlocal_closing_free()
    self.default_convert(mod_x, 2)
    self.assertEqual(read_x(), 2)


if __name__ == '__main__':
  test.main()

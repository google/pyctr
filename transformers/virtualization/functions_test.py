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
"""Tests for function converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from absl.testing import absltest as test
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.overloads.testing import call_swapping
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import variables


class FunctionConversionTest(test.TestCase):

  def default_convert(self, func, *args, **keywords):
    converted_func = conversion.convert(func, py_defaults, [functions])
    return converted_func(*args, **keywords)

  def test_external_function_calls(self):

    def foo(x):
      return x

    def test_fn(x):
      return foo(x)

    for x in range(-5, 5):
      self.assertEqual(self.default_convert(test_fn, x), test_fn(x))

  def test_nested_function_calls(self):

    def outer_fn(x):

      def inner_fn(y):
        return 1 + y

      return inner_fn(x)

    for x in range(-5, 5):
      self.assertEqual(self.default_convert(outer_fn, x), outer_fn(x))

  def test_higher_order_function(self):

    def test_fn(func, x):
      return func(x)

    lambda_func = lambda x: x + 1
    self.assertEqual(
        self.default_convert(test_fn, lambda_func, 3), test_fn(lambda_func, 3))

  def test_keywords(self):

    def foo(x, keyword=True):
      return x if keyword else x + 1

    def test_fn(x, keyword):
      return foo(x, keyword=keyword)

    self.assertEqual(
        self.default_convert(test_fn, 1, keyword=True), test_fn(
            1, keyword=True))
    self.assertEqual(
        self.default_convert(test_fn, 1, keyword=False),
        test_fn(1, keyword=False))

  def test_nullary_function(self):

    def foo():
      return None

    def test_fn():
      return foo()

    self.assertEqual(self.default_convert(test_fn), test_fn())

  def test_attributes(self):

    class TestClass(object):

      def foo(self, x):
        return x

    def test_fn(tc, x):
      return tc.foo(x)

    tc = TestClass()
    self.assertEqual(self.default_convert(test_fn, tc, 3), test_fn(tc, 3))

  def test_known_function_swapped(self):

    # add is overloaded in call_swapping to return x + y + 1 for ints
    def test_fn(x, y):
      return call_swapping.add(x, y)

    converted_func = conversion.convert(test_fn, call_swapping, [functions])
    self.assertEqual(converted_func(1, 2), test_fn(1, 2) + 1)
    self.assertEqual(
        converted_func('hello ', 'world'), test_fn('hello ', 'world'))

  def test_unknown_function_not_swapped(self):

    # no overload for adder exists in call_swapping
    def adder(x, y):
      return x + y

    def test_fn(x, y):
      return adder(x, y)

    converted_func = conversion.convert(test_fn, call_swapping, [functions])
    self.assertEqual(converted_func(1, 2), test_fn(1, 2))
    self.assertEqual(
        converted_func('hello ', 'world'), test_fn('hello ', 'world'))

  def test_overload_not_staged(self):
    """Calls on the overload module are not themselves overloaded."""

    # After conversion this function will contain function calls for variable
    # virtualization, we do not want these to be affected by function call
    # virtualization.
    def test_fn(y):
      x = 1 + y
      return x

    # Custom overloads that blow up on call virtualization.
    class MyOverloads(object):

      # There is no call in the test_fn that should be virtualized this ensures
      # that an exception is thrown if call virtualization is attempted.
      def call(self, func, args, keywords):  # pylint: disable=unused-argument
        self.fail('No call should be virtualized in this test.')

      def init(self, name):
        return py_defaults.init(name)

      def assign(self, lhs, rhs):
        return py_defaults.assign(lhs, rhs)

      def read(self, var):
        return py_defaults.read(var)

    overloads = MyOverloads()

    converted_func = conversion.convert(test_fn, overloads,
                                        [variables, functions])
    self.assertEqual(converted_func(5), 6)


if __name__ == '__main__':
  test.main()

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
"""Tests for variables converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.overloads.testing import dictionary_variables
from pyctr.overloads.testing import increment_var_reads_logic as inc_reads
from pyctr.transformers.virtualization import variables


a_global_var = 0
another_global_var = 0


def func_with_globals():
  global a_global_var
  a_global_var = a_global_var + 1
  return a_global_var


def inner_func_with_globals():
  x = 1

  def inner_func():
    global another_global_var
    another_global_var = another_global_var + x
    return another_global_var

  return inner_func


def assign_read_no_params():
  a = 1
  b = a + 1
  return b


def assign_read_with_params(x):
  y = x + 1
  return y - x


class VariablesConversionTest(parameterized.TestCase):

  def default_convert(self, func, *args):
    converted_func = conversion.convert(func, py_defaults, [variables])
    return converted_func(*args)

  def test_noop(self):
    self.assertEqual(
        self.default_convert(assign_read_no_params), assign_read_no_params())

  def test_noop_dictionary_variables(self):
    converted_assign_read_no_params = conversion.convert(
        assign_read_no_params, dictionary_variables, [variables])
    self.assertEqual(converted_assign_read_no_params(), assign_read_no_params())

  def test_noop_with_params(self):
    self.assertEqual(
        self.default_convert(assign_read_with_params, 5),
        assign_read_with_params(5))

  def test_noop_with_params_dictionary_variables(self):
    converted_assign_read_with_params = conversion.convert(
        assign_read_with_params, dictionary_variables, [variables])
    self.assertEqual(
        converted_assign_read_with_params(5), assign_read_with_params(5))

  def test_add_one(self):
    converted_assign_read_no_params = conversion.convert(
        assign_read_no_params, inc_reads, [variables])
    self.assertEqual(converted_assign_read_no_params(),
                     4)  # All reads add one to the value

  def test_add_one_with_params(self):
    converted_assign_read_with_params = conversion.convert(
        assign_read_with_params, inc_reads, [variables])
    self.assertEqual(converted_assign_read_with_params(5), 2)

  def test_read(self):

    def enclosing_func():
      x = 1

      def inner_func():
        return x

      return inner_func()

    self.assertEqual(self.default_convert(enclosing_func), enclosing_func())

  def test_read_convert_inner(self):

    def enclosing_func():
      x = 1

      def inner_func():
        return x

      return inner_func

    inner = enclosing_func()
    self.assertEqual(self.default_convert(inner), inner())

  def test_vars_defined_after_inner(self):

    def enclosing_func():
      x = 1

      def inner_func():
        return x + y

      y = 2
      return inner_func()

    self.assertEqual(self.default_convert(enclosing_func), enclosing_func())

  def test_if_chains(self):

    def func(cond):
      if cond:
        x = 1

      return x

    self.assertEqual(self.default_convert(func, True), func(True))
    with self.assertRaises(py_defaults.PyctUnboundLocalError):
      self.default_convert(func, False)

  def test_nested_funcs(self):

    def outer_func():
      a = 1

      def inner_func():
        return a

      return inner_func()

    self.assertEqual(self.default_convert(outer_func), outer_func())

  def test_before_and_after(self):

    def vars_before_and_after():
      x = 1

      def inner_func():
        return x, y

      y = 2
      return inner_func()

    self.assertEqual(
        self.default_convert(vars_before_and_after), vars_before_and_after())

  def test_global_outer(self):
    self.assertEqual(self.default_convert(func_with_globals), 1)
    self.assertEqual(a_global_var, 1)

  def test_global_inner(self):
    global_inner = inner_func_with_globals()
    self.assertEqual(self.default_convert(global_inner), 1)
    self.assertEqual(another_global_var, 1)

  def test_in_loop(self):

    def func(c1, c2):
      i = 0
      while i < 5:
        if c1(i):
          x = i

        if c2(i):
          x = x + 1

        i = i + 1
      return x

    true_lambda = lambda _: True
    false_lambda = lambda _: False

    self.assertEqual(
        self.default_convert(func, true_lambda, false_lambda),
        func(true_lambda, false_lambda))

    is_even = lambda x: x % 2 == 0
    is_gt_1 = lambda x: x > 1

    self.assertEqual(
        self.default_convert(func, is_even, is_gt_1), func(is_even, is_gt_1))

    with self.assertRaises(py_defaults.PyctUnboundLocalError):
      self.default_convert(func, false_lambda, true_lambda)

    with self.assertRaises(py_defaults.PyctUnboundLocalError):
      self.default_convert(func, false_lambda, false_lambda)

  @parameterized.parameters(
      (-1),
      (0),
      (1),
      (5),
  )
  def test_for_loop(self, x):

    def test_fn(n):
      sum_ = 0

      for i in range(n):
        sum_ = sum_ + i

      return sum_

    converted_fn = conversion.convert(test_fn, py_defaults, [variables])
    self.assertEqual(converted_fn(x), test_fn(x))

  @parameterized.named_parameters(
      {
          'testcase_name': 'empty_list',
          'iter_': []
      },
      {
          'testcase_name': 'singleton',
          'iter_': [(1, 2)]
      },
      {
          'testcase_name': 'cons',
          'iter_': [(1, 2), (3, 4)]
      },
  )
  def test_for_multiple_targets(self, iter_):

    def test_fn(iter_):
      res = []

      for x, y in iter_:
        res.append((x, y))

      return res

    converted_fn = conversion.convert(test_fn, py_defaults, [variables])
    self.assertEqual(converted_fn(iter_), test_fn(iter_))

  @parameterized.parameters(
      (1),
      (5),
  )
  def test_for_loop_return_target(self, x):

    def test_fn(n):
      sum_ = 0

      for i in range(n):
        sum_ = sum_ + i

      return sum_ + i  # pylint: disable=undefined-loop-variable

    converted_fn = conversion.convert(test_fn, py_defaults, [variables])
    self.assertEqual(converted_fn(x), test_fn(x))

  @parameterized.parameters(
      (-1),
      (0),
  )
  def test_for_loop_return_target_le_0(self, x):

    def test_fn(n):
      sum_ = 0

      for i in range(n):
        sum_ = sum_ + i

      return sum_ + i  # pylint: disable=undefined-loop-variable

    converted_fn = conversion.convert(test_fn, py_defaults, [variables])
    with self.assertRaises(py_defaults.PyctUnboundLocalError):
      converted_fn(x)


if __name__ == '__main__':
  test.main()

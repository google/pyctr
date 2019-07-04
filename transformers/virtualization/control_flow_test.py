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
"""Tests for control_flow converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.overloads.testing import dictionary_variables
from pyctr.overloads.testing import reverse_conditional_logic as rev_cond
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import variables


def check_cond(i):
  v = []

  if i < 5:
    v.append(1)
  else:
    v.append(2)
  return v


class ControlFlowTest(parameterized.TestCase):

  def test_noop_cond(self):
    converted_check_cond = conversion.convert(check_cond, py_defaults,
                                              [control_flow])

    self.assertEqual(converted_check_cond(1), check_cond(1))
    self.assertEqual(converted_check_cond(5), check_cond(5))

  def test_noop_py_dictionary_variables(self):
    converted_check_cond = conversion.convert(check_cond, dictionary_variables,
                                              [control_flow])
    self.assertEqual(converted_check_cond(1), check_cond(1))
    self.assertEqual(converted_check_cond(5), check_cond(5))

  def test_if_swap(self):
    converted_check_cond = conversion.convert(check_cond, rev_cond,
                                              [control_flow])
    self.assertEqual(converted_check_cond(1), check_cond(5))
    self.assertEqual(converted_check_cond(5), check_cond(1))

  def test_if_no_else(self):

    def test_fn(n):
      v = []

      if n > 0:
        v.append(n)

      return v

    converted_fn = conversion.convert(test_fn, py_defaults, [control_flow])
    for i in [0, 1]:
      self.assertEqual(converted_fn(i), test_fn(i))

  def test_noop_while(self):

    def test_fn(n):
      res = []
      i = 0

      while i < n:
        res.append(i)
        i = i + 1

      return res

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
    self.assertEqual(converted_fn(5), test_fn(5))

  def test_conds_in_while(self):

    def test_fn(n):
      res = []
      i = 0

      while i < n:
        if i % 2 == 0:
          res.append(i)
        i = i + 1

      return res

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
    self.assertEqual(converted_fn(5), test_fn(5))

  @parameterized.parameters(
      (-1),
      (0),
      (1),
      (5),
  )
  def test_for_noop(self, x):

    def test_fn(n):
      res = []

      for i in range(n):
        res.append(i)

      return res

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
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
  def test_for_parameterized_noop(self, iter_):

    def test_fn(iter_):
      res = []

      for x, y in iter_:
        res.append((x, y))

      return res

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
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

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
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

    converted_fn = conversion.convert(test_fn, py_defaults,
                                      [variables, control_flow])
    with self.assertRaises(py_defaults.PyctUnboundLocalError):
      converted_fn(x)


if __name__ == '__main__':
  test.main()

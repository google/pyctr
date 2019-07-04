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
"""Tests for logical_ops converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.overloads.testing import reverse_logical_ops
from pyctr.transformers.virtualization import logical_ops


class LogicalOpsTest(parameterized.TestCase):

  def default_convert(self, f):
    return conversion.convert(f, py_defaults, [logical_ops])

  def reverse_convert(self, f):
    return conversion.convert(f, reverse_logical_ops, [logical_ops])

  @parameterized.named_parameters(('TT', True, True), ('TF', True, False),
                                  ('FT', False, True), ('FF', False, False))
  def test_and(self, a, b):

    def test_fn(x, y):
      return x and y

    converted_fn = self.default_convert(test_fn)
    self.assertEqual(test_fn(a, b), converted_fn(a, b))

  def test_and_lazy(self):

    def foo():
      raise ValueError()

    def test_fn(x):
      return x and foo()

    converted_fn = self.default_convert(test_fn)

    self.assertFalse(converted_fn(False))
    with self.assertRaises(ValueError):
      converted_fn(True)

  @parameterized.named_parameters(
      ('TTT', True, True, True), ('TTF', True, True, False),
      ('TFT', True, False, True), ('TFF', True, False, False),
      ('FTT', False, True, True), ('FTF', False, True, False),
      ('FFT', False, False, True), ('FFF', False, False, False))
  def test_and_multiple(self, a, b, c):

    def test_fn(x, y, z):
      return x and y and z

    converted_fn = self.default_convert(test_fn)
    self.assertEqual(test_fn(a, b, c), converted_fn(a, b, c))

  def test_and_lazy_multiple(self):

    def foo():
      raise ValueError()

    def test_fn(x, y):
      return x and y and foo()

    converted_fn = self.default_convert(test_fn)

    self.assertFalse(converted_fn(True, False))
    self.assertFalse(converted_fn(False, True))
    self.assertFalse(converted_fn(False, False))
    with self.assertRaises(ValueError):
      converted_fn(True, True)

  @parameterized.named_parameters(('TT', True, True), ('TF', True, False),
                                  ('FT', False, True), ('FF', False, False))
  def test_or(self, a, b):

    def test_fn(x, y):
      return x or y

    converted_fn = self.default_convert(test_fn)
    self.assertEqual(test_fn(a, b), converted_fn(a, b))

  def test_or_lazy(self):

    def foo():
      raise ValueError()

    def test_fn(x):
      return x or foo()

    converted_fn = self.default_convert(test_fn)

    self.assertTrue(converted_fn(True))
    with self.assertRaises(ValueError):
      converted_fn(False)

  @parameterized.named_parameters(
      ('TTT', True, True, True), ('TTF', True, True, False),
      ('TFT', True, False, True), ('TFF', True, False, False),
      ('FTT', False, True, True), ('FTF', False, True, False),
      ('FFT', False, False, True), ('FFF', False, False, False))
  def test_or_multiple(self, a, b, c):

    def test_fn(x, y, z):
      return x or y or z

    converted_fn = self.default_convert(test_fn)
    self.assertEqual(test_fn(a, b, c), converted_fn(a, b, c))

  def test_or_lazy_multiple(self):

    def foo():
      raise ValueError()

    def test_fn(x, y):
      return x or y or foo()

    converted_fn = self.default_convert(test_fn)

    self.assertTrue(converted_fn(True, True))
    self.assertTrue(converted_fn(True, False))
    self.assertTrue(converted_fn(False, True))
    with self.assertRaises(ValueError):
      converted_fn(False, False)

  def test_Not(self):

    def test_fn(x):
      return not x

    converted_fn = self.default_convert(test_fn)
    self.assertTrue(converted_fn(False))
    self.assertFalse(converted_fn(True))

  @parameterized.named_parameters(('TT', True, True), ('TF', True, False),
                                  ('FT', False, True), ('FF', False, False))
  def test_and_as_or(self, a, b):

    def test_fn(x, y):
      return x and y

    def truth_fn(x, y):
      return x or y

    converted_fn = self.reverse_convert(test_fn)
    self.assertEqual(truth_fn(a, b), converted_fn(a, b))

  @parameterized.named_parameters(('TT', True, True), ('TF', True, False),
                                  ('FT', False, True), ('FF', False, False))
  def test_or_as_and(self, a, b):

    def test_fn(x, y):
      return x or y

    def truth_fn(x, y):
      return x and y

    converted_fn = self.reverse_convert(test_fn)
    self.assertEqual(truth_fn(a, b), converted_fn(a, b))

  def test_Not_reverse(self):

    def test_fn(x):
      return not x

    converted_fn = self.reverse_convert(test_fn)
    self.assertTrue(converted_fn(True))
    self.assertFalse(converted_fn(False))


if __name__ == '__main__':
  test.main()

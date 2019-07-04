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
"""Tests for conversion module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from pyctr.api import conversion
from pyctr.overloads import py_defaults
from pyctr.overloads.testing import dictionary_variables


def check_cond(i):
  v = []

  if i < 5:
    v.append(1)
  else:
    v.append(2)
  return v


class ConversionTest(test.TestCase):

  def test_noop(self):
    self.assertListEqual(check_cond(1), [1])
    self.assertListEqual(check_cond(5), [2])
    converted_check_cond = conversion.convert(check_cond, py_defaults,
                                              [])
    self.assertListEqual(converted_check_cond(1), [1])
    self.assertListEqual(check_cond(5), [2])

  def test_noop_dictionary_variables(self):
    self.assertListEqual(check_cond(1), [1])
    self.assertListEqual(check_cond(5), [2])
    converted_check_cond = conversion.convert(check_cond, dictionary_variables,
                                              [])
    self.assertListEqual(converted_check_cond(1), [1])
    self.assertListEqual(check_cond(5), [2])


if __name__ == '__main__':
  test.main()

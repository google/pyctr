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
"""Tests for tf overload module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.examples.tf import tf as tf_
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import variables

import tensorflow as tf


class TfTest(parameterized.TestCase):

  def _convert(self, f, transformers):
    return conversion.convert(f, tf_, transformers)

  def convert_tf(self, f):
    return self._convert(f, [variables, control_flow])

  def assertEqualUnconvertedConverted(self, test_fn, converted_fn, params):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        unconverted_result = test_fn(*params)
        converted_result = converted_fn(*[tf.constant(x) for x in params])
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters((-1), (1))
  def test_if_basic(self, n):

    def test_fn(n):
      a = 0
      b = 0
      if n > 0:
        a = n
      else:
        b = n
      return a, b

    converted_fn = conversion.convert(test_fn, tf_,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(n))
        unconverted_result = test_fn(n)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters(
      {
          'x': 1,
          'y': 1
      },
      {
          'x': 1,
          'y': -1
      },
      {
          'x': -2,
          'y': 2
      },
      {
          'x': -2,
          'y': -2
      },
  )
  def test_nested_if(self, x, y):

    def test_fn(x, y):
      a = 0
      b = 0
      c = 0
      d = 0
      if x > 0:
        if y > 0:
          a = x
        else:
          b = y
      else:
        if y > 0:
          c = x
        else:
          d = y
      return a, b, c, d

    converted_fn = conversion.convert(test_fn, tf_,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(x), tf.constant(y))
        unconverted_result = test_fn(x, y)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters((5), (3), (2), (1), (-5), (-3), (-2), (0))
  def test_very_nested_if(self, x):

    def test_fn(x):
      a = 0
      b = 0
      c = 0
      d = 0
      e = 0
      f = 0
      g = 0
      h = 0
      then_branch = 0
      else_branch = 0

      if x > 0:
        if x > 2:
          if x > 4:
            a = 1
          else:
            b = 1
        else:
          if x > 1:
            c = 1
          else:
            d = 1
        then_branch = 1
      else:
        if x < -2:
          if x < -4:
            e = 1
          else:
            f = 1
        else:
          if x < -1:
            g = 1
          else:
            h = 1
        else_branch = 1
      return a, b, c, d, e, f, g, h, then_branch, else_branch

    converted_fn = conversion.convert(test_fn, tf_,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(x))
        unconverted_result = test_fn(x)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters((-1), (1))
  def test_sequential_ifs(self, x):

    def test_fn(x):
      a = 0
      b = 0

      if x > 0:
        a = 1
      else:
        b = 1

      if x < 0:
        a = 2
      else:
        b = 2
      return a, b

    converted_fn = conversion.convert(test_fn, tf_,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(x))
        unconverted_result = test_fn(x)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters((-1), (1), (5))
  def test_while_basic(self, x):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s = s + i
        i = i + 1
      return s, i, n

    converted_fn = conversion.convert(test_fn, tf_,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(x))
        unconverted_result = test_fn(x)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  @parameterized.parameters((-1), (1), (5))
  def test_while_nested(self, x):

    def test_fn(n):
      i = 0
      j = 0
      s = 0
      while i < n:
        while j < i:
          j = j + 3
        u = i + j  # 'u' is not defined within the inner loop
        s = s + u
        i = i + 1
        j = 0
      return s, i, j, n

    converted_fn = conversion.convert(test_fn, tf_, [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(tf.constant(x))
        unconverted_result = test_fn(x)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  # TODO(jmd1011): Handle empty list correctly. Right now, TF treats [] as type
  # float32.
  @parameterized.named_parameters(
      {
          'testcase_name': '[]',
          'l': []
      },
      {
          'testcase_name': '[1, 3]',
          'l': [1., 3.]
      },
      {
          'testcase_name': '[1, 2, 3]',
          'l': [1., 2., 3.]
      },
  )
  def test_for_single_output(self, l):

    def test_fn(l):
      s = 0.
      for e in l:
        s = s + e
      return s

    converted_fn = self.convert_tf(test_fn)
    self.assertEqualUnconvertedConverted(test_fn, converted_fn, (l,))


if __name__ == '__main__':
  test.main()

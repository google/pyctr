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
"""Tests for numpy_to_tf overload module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
import numpy as np
from pyctr.api import conversion
from pyctr.examples.numpy import numpy_to_tf
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import variables
import tensorflow as tf


class NumPyTest(parameterized.TestCase):

  def test_basic(self):

    def test_fn(n):
      i = np.zeros((2, 3))
      s = np.zeros((2, 3))
      while np.amax(i) < n:
        if np.amax(i) > np.amax(s):
          u = i - s
          j = np.zeros(())
        else:
          u = i + s
          j = np.ones(())
        s = s + u + j
        i = i + 1
      return s[1][1], i[1][1]

    converted_fn = conversion.convert(test_fn, numpy_to_tf,
                                      [variables, control_flow, functions])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(3)
        unconverted_result = test_fn(3)
        self.assertEqual(sess.run(converted_result), unconverted_result)


if __name__ == '__main__':
  test.main()

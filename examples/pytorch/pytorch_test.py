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
from pyctr.examples.pytorch import pytorch
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import variables
import tensorflow as tf
import torch


class PyTorchTest(parameterized.TestCase):

  @parameterized.parameters((-1), (1), (5))
  def test_basic_control_flow(self, x):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        if i > s:
          u = i - s
          j = 0
        else:
          u = i + s
          j = 1
        s = s + u + j
        i = i + 1
      return s, i

    converted_fn = conversion.convert(test_fn, pytorch,
                                      [variables, control_flow])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn(torch.tensor(x, dtype=torch.int32))
        unconverted_result = test_fn(x)
        self.assertEqual(sess.run(converted_result), unconverted_result)

  # TODO(mdanatg): Add tests for individual function conversions.

  def test_api_rewrite(self):

    # TODO(b/129431400): Remove this alias.
    conv1d = torch.nn.functional.conv1d

    def test_fn():
      value = torch.rand(7, 3, 25)
      filters = torch.rand(11, 3, 5)
      return conv1d(value, filters)

    converted_fn = conversion.convert(test_fn, pytorch,
                                      [variables, control_flow, functions])

    with tf.Graph().as_default():
      with tf.Session() as sess:
        converted_result = converted_fn()
        unconverted_result = test_fn()
        self.assertEqual(
            sess.run(converted_result).shape,
            unconverted_result.numpy().shape)


if __name__ == '__main__':
  test.main()

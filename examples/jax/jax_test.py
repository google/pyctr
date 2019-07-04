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
"""Tests for jax overload module. Based on tests found in the JAX code base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
import jax
from jax import lax
from pyctr.api import conversion
from pyctr.examples.jax import jax as jax_
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import variables


class JaxTest(parameterized.TestCase):

  @parameterized.parameters((-1), (1))
  def test_if_basic(self, x):

    def test_fn(n):
      a = 0
      b = 0
      if n > 0:
        a = n
      else:
        b = n
      return a, b

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jax_converted = jax.jit(converted_fn)
    converted_result = jax_converted(x)
    unconverted_result = test_fn(x)
    self.assertEqual(converted_result, unconverted_result)

  @parameterized.parameters((1, 1), (1, -1), (-2, 2), (-2, -2))
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

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jax_converted = jax.jit(converted_fn)
    converted_result = jax_converted(x, y)
    unconverted_result = test_fn(x, y)
    self.assertEqual(converted_result, unconverted_result)

  @parameterized.parameters((5), (3), (2), (1), -(5), -(3), -(2), (0))
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

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jax_converted = jax.jit(converted_fn)
    converted_result = jax_converted(x)
    unconverted_result = test_fn(x)
    self.assertEqual(converted_result, unconverted_result)

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

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jax_converted = jax.jit(converted_fn)
    converted_result = jax_converted(x)
    unconverted_result = test_fn(x)
    self.assertEqual(converted_result, unconverted_result)

  @parameterized.parameters((1), (3), (6))
  def test_nested_cond(self, x):

    def fun(x):
      res = 0
      if x < 2:
        res = lax.mul(2, x)
      else:
        if x < 5:
          res = lax.mul(3, x)
        else:
          res = lax.mul(4, x)
      return res

    @jax.api.jit
    def cfun(x):
      def inner_cond(x):
        return lax.cond(
            lax.lt(x, 5),
            x,
            lambda x: lax.mul(3, x),
            4,
            lambda y: lax.mul(y, x),
        )

      return lax.cond(lax.lt(x, 2), x, lambda x: lax.mul(2, x), x, inner_cond)

    converted_fn = conversion.convert(fun, jax_, [variables, control_flow])
    self.assertEqual(cfun(x), converted_fn(x))

  def test_jit_after_conversion_cond(self):

    def f(x):
      res = 0
      if x < 3:
        res = 3. * x**2
      else:
        res = -4. * x
      return res

    converted_fn = conversion.convert(f, jax_, [variables, control_flow])
    self.assertEqual(f(2), converted_fn(2))

    jitted_fn = jax.api.jit(converted_fn)
    self.assertEqual(f(2), jitted_fn(2))

  @parameterized.parameters((2), (2), (2), (3))
  def test_while_basic(self, i):
    limit = 10

    def test_fn(init):
      count = 0
      while init < limit:
        init = init + 1
        count = count + 1
      return count

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jitted_fn = jax.api.jit(converted_fn)
    self.assertEqual(test_fn(i), limit - i)
    self.assertEqual(test_fn(i), converted_fn(i))
    self.assertEqual(test_fn(i), jitted_fn(i))

  @parameterized.parameters((2), (3), (3), (4))
  def test_while_nested(self, i):

    def test_fn(n):
      count = 0
      i = 0
      while i < n:
        i = i + 1

        j = 0
        while j <= i:
          j = j + 1
          count = count + 1
      return count

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    self.assertEqual(test_fn(2), converted_fn(2))

    jitted_fn = jax.api.jit(converted_fn)
    self.assertEqual(test_fn(i), jitted_fn(i))

  def test_while_with_closure(self):

    effect = [False]

    def get_effect():
      return effect[0]

    def set_effect(b):
      effect[0] = b

    def test_fn(init, local_limit, inc):
      count = 0
      while init < local_limit:
        set_effect(True)
        init = init + 1
        count = count + inc
      return count

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jitted_fn = jax.api.jit(converted_fn)

    limit = 10

    for i in (2, 2):
      expected = limit - i

      unconverted_result = test_fn(i, limit, 1)
      converted_result = converted_fn(i, limit, 1)
      jitted_result = jitted_fn(i, limit, 1)
      self.assertEqual(expected, unconverted_result)
      self.assertEqual(expected, converted_result)
      self.assertEqual(expected, jitted_result)
      assert get_effect()
      set_effect(False)  # reset locally mutable effect

    self.assertEqual(jitted_fn(2, limit, 1), limit - 2)
    self.assertEqual(jitted_fn(3, limit, 1), limit - 3)
    assert not get_effect()

  @parameterized.parameters((2), (3), (4))
  def test_for_loop_basic(self, n):

    def test_jax(num):

      def body_fun(i, tot):
        return lax.add(tot, i)

      return lax.fori_loop(0, num, body_fun, 0)

    def test_fn(n):
      s = 0
      for i in n:
        s = s + i
      return s

    converted_fn = conversion.convert(test_fn, jax_, [variables, control_flow])
    jitted = jax.api.jit(converted_fn)
    self.assertEqual(jitted(jax.numpy.arange(n)), test_fn(range(n)))
    self.assertEqual(jitted(jax.numpy.arange(n)), test_jax(n))


if __name__ == '__main__':
  test.main()

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
"""Tests for parsing module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import textwrap

from absl.testing import absltest as test
import gast
from pyctr.core import parsing


class ParsingTest(test.TestCase):

  def test_parse_entity(self):

    def f(x):
      return x + 1

    mod, _ = parsing.parse_entity(f)
    self.assertEqual('f', mod.body[0].name)

  def test_parse_str(self):
    mod = parsing.parse_str(
        textwrap.dedent("""
            def f(x):
              return x + 1
    """))
    self.assertEqual('f', mod.body[0].name)

  def test_parse_comments(self):

    def f():
# unindented comment
      pass

    with self.assertRaises(ValueError):
      parsing.parse_entity(f)

  def test_parse_multiline_strings(self):

    def f():
      print("""
some
multiline
string""")

    with self.assertRaises(ValueError):
      parsing.parse_entity(f)

  def test_parse_expression(self):
    node = parsing.parse_expression('a.b')
    self.assertEqual('a', node.value.id)
    self.assertEqual('b', node.attr)

  def test_parsing_compile_idempotent(self):

    def test_fn(x):
      a = True
      b = ''
      if a:
        b = x + 1
      return b

    self.assertEqual(
        textwrap.dedent(inspect.getsource(test_fn)),
        inspect.getsource(
            parsing.ast_to_object(
                parsing.parse_entity(test_fn)[0].body[0])[0].test_fn))

  def test_ast_to_source(self):
    node = gast.If(
        test=gast.Num(1),
        body=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Name('b', gast.Load(), None))
        ],
        orelse=[
            gast.Assign(
                targets=[gast.Name('a', gast.Store(), None)],
                value=gast.Str('c'))
        ])

    source = parsing.ast_to_source(node, indentation='  ')
    self.assertEqual(
        textwrap.dedent("""
            if 1:
              a = b
            else:
              a = 'c'
        """).strip(), source.strip())

  def test_ast_to_object(self):
    node = gast.FunctionDef(
        name='f',
        args=gast.arguments(
            args=[gast.Name('a', gast.Param(), None)],
            vararg=None,
            kwonlyargs=[],
            kwarg=None,
            defaults=[],
            kw_defaults=[]),
        body=[
            gast.Return(
                gast.BinOp(
                    op=gast.Add(),
                    left=gast.Name('a', gast.Load(), None),
                    right=gast.Num(1)))
        ],
        decorator_list=[],
        returns=None)

    module, source = parsing.ast_to_object(node)

    expected_source = """
      def f(a):
        return a + 1
    """
    self.assertEqual(textwrap.dedent(expected_source).strip(), source.strip())
    self.assertEqual(2, module.f(1))
    with open(module.__file__, 'r') as temp_output:
      self.assertEqual(
          textwrap.dedent(expected_source).strip(),
          temp_output.read().strip())


if __name__ == '__main__':
  test.main()

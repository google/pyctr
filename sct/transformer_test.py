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
"""Tests for templates module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
import gast
from pyctr.core import anno
from pyctr.core import parsing
from pyctr.sct import transformer


class TransformerTest(test.TestCase):

  def _simple_source_info(self):
    return transformer.EntityInfo(
        source_code=None,
        source_file=None,
        namespace=None,
        arg_values=None,
        arg_types=None,
        owner_type=None)

  def test_entity_scope_tracking(self):

    class TestTransformer(transformer.Base):

      # The choice of note to assign to is arbitrary. Using Assign because it's
      # easy to find in the tree.
      def visit_Assign(self, node):
        anno.setanno(node, 'enclosing_entities', self.enclosing_entities)
        return self.generic_visit(node)

      # This will show up in the lambda function.
      def visit_BinOp(self, node):
        anno.setanno(node, 'enclosing_entities', self.enclosing_entities)
        return self.generic_visit(node)

    tr = TestTransformer(self._simple_source_info())

    def test_function():
      a = 0

      class TestClass(object):

        def test_method(self):
          b = 0

          def inner_function(x):
            c = 0
            d = lambda y: (x + y)
            return c, d

          return b, inner_function

      return a, TestClass

    node, _ = parsing.parse_entity(test_function)
    node = tr.visit(node)

    test_function_node = node.body[0]
    test_class = test_function_node.body[1]
    test_method = test_class.body[0]
    inner_function = test_method.body[1]
    lambda_node = inner_function.body[1].value

    a = test_function_node.body[0]
    b = test_method.body[0]
    c = inner_function.body[0]
    lambda_expr = lambda_node.body

    self.assertEqual((test_function_node,), anno.getanno(
        a, 'enclosing_entities'))
    self.assertEqual((test_function_node, test_class, test_method),
                     anno.getanno(b, 'enclosing_entities'))
    self.assertEqual(
        (test_function_node, test_class, test_method, inner_function),
        anno.getanno(c, 'enclosing_entities'))
    self.assertEqual((test_function_node, test_class, test_method,
                      inner_function, lambda_node),
                     anno.getanno(lambda_expr, 'enclosing_entities'))

  def assertSameAnno(self, first, second, key):
    self.assertIs(anno.getanno(first, key), anno.getanno(second, key))

  def assertDifferentAnno(self, first, second, key):
    self.assertIsNot(anno.getanno(first, key), anno.getanno(second, key))

  def test_state_tracking(self):

    class LoopState(object):
      pass

    class CondState(object):
      pass

    class TestTransformer(transformer.Base):

      def visit(self, node):
        anno.setanno(node, 'loop_state', self.state[LoopState].value)
        anno.setanno(node, 'cond_state', self.state[CondState].value)
        return super(TestTransformer, self).visit(node)

      def visit_While(self, node):
        self.state[LoopState].enter()
        node = self.generic_visit(node)
        self.state[LoopState].exit()
        return node

      def visit_If(self, node):
        self.state[CondState].enter()
        node = self.generic_visit(node)
        self.state[CondState].exit()
        return node

    tr = TestTransformer(self._simple_source_info())

    def test_function(a):
      a = 1
      while a:
        _ = 'a'
        if a > 2:
          _ = 'b'
          while True:
            raise '1'
        if a > 3:
          _ = 'c'
          while True:
            raise '1'

    node, _ = parsing.parse_entity(test_function)
    node = tr.visit(node)

    fn_body = node.body[0].body
    outer_while_body = fn_body[1].body
    self.assertSameAnno(fn_body[0], outer_while_body[0], 'cond_state')
    self.assertDifferentAnno(fn_body[0], outer_while_body[0], 'loop_state')

    first_if_body = outer_while_body[1].body
    self.assertDifferentAnno(outer_while_body[0], first_if_body[0],
                             'cond_state')
    self.assertSameAnno(outer_while_body[0], first_if_body[0], 'loop_state')

    first_inner_while_body = first_if_body[1].body
    self.assertSameAnno(first_if_body[0], first_inner_while_body[0],
                        'cond_state')
    self.assertDifferentAnno(first_if_body[0], first_inner_while_body[0],
                             'loop_state')

    second_if_body = outer_while_body[2].body
    self.assertDifferentAnno(first_if_body[0], second_if_body[0], 'cond_state')
    self.assertSameAnno(first_if_body[0], second_if_body[0], 'loop_state')

    second_inner_while_body = second_if_body[1].body
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'cond_state')
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'loop_state')

  def test_visit_block_postprocessing(self):

    class TestTransformer(transformer.Base):

      def _process_body_item(self, node):
        if isinstance(node, gast.Assign) and (node.value.id == 'y'):
          if_node = gast.If(gast.Name('x', gast.Load(), None), [node], [])
          return if_node, if_node.body
        return node, None

      def visit_FunctionDef(self, node):
        node.body = self.visit_block(
            node.body, after_visit=self._process_body_item)
        return node

    def test_function(x, y):
      z = x
      z = y
      return z

    tr = TestTransformer(self._simple_source_info())

    node, _ = parsing.parse_entity(test_function)
    node = tr.visit(node)
    node = node.body[0]

    self.assertLen(node.body, 2)
    self.assertIsInstance(node.body[0], gast.Assign)
    self.assertIsInstance(node.body[1], gast.If)
    self.assertIsInstance(node.body[1].body[0], gast.Assign)
    self.assertIsInstance(node.body[1].body[1], gast.Return)

  def test_robust_error_on_list_visit(self):

    class BrokenTransformer(transformer.Base):

      def visit_If(self, node):
        # This is broken because visit expects a single node, not a list, and
        # the body of an if is a list.
        # Importantly, the default error handling in visit also expects a single
        # node.  Therefore, mistakes like this need to trigger a type error
        # before the visit called here installs its error handler.
        # That type error can then be caught by the enclosing call to visit,
        # and correctly blame the If node.
        self.visit(node.body)
        return node

    def test_function(x):
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_source_info())

    node, _ = parsing.parse_entity(test_function)
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    expected_message = r'expected "ast.AST", got "\<(type|class) \'list\'\>"'
    self.assertRegexpMatches(obtained_message, expected_message)

  def test_robust_error_on_ast_corruption(self):
    # A child class should not be able to be so broken that it causes the error
    # handling in `transformer.Base` to raise an exception.  Why not?  Because
    # then the original error location is dropped, and an error handler higher
    # up in the call stack gives misleading information.

    # Here we test that the error handling in `visit` completes, and blames the
    # correct original exception, even if the AST gets corrupted.

    class NotANode(object):
      pass

    class BrokenTransformer(transformer.Base):

      def visit_If(self, node):
        node.body = NotANode()
        raise ValueError('I blew up')

    def test_function(x):
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_source_info())

    node, _ = parsing.parse_entity(test_function)
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    # The message should reference the exception actually raised, not anything
    # from the exception handler.
    expected_substring = 'I blew up'
    self.assertIn(expected_substring, obtained_message, obtained_message)


if __name__ == '__main__':
  test.main()

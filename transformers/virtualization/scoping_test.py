# python3

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

from absl.testing import absltest as test
from pyctr.core import naming
from pyctr.core import parsing
from pyctr.sct import transformer
from pyctr.transformers.virtualization import scoping

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


def assign_read_with_params(x):
  y = x + 1
  return y - x


class ScopingTest(test.TestCase):

  def get_scopes(self, func):
    source, _ = parsing.parse_entity(func)
    entity_info = transformer.EntityInfo(
        source_code=source,
        source_file='<fragment>',
        namespace={},
        arg_values=None,
        arg_types={},
        owner_type=None)

    namer = naming.Namer(entity_info.namespace)
    ctx = transformer.EntityContext(namer, entity_info)
    scope_transformer = scoping.ScopeTransformer(ctx)
    scope_transformer.visit(source)
    return scope_transformer.scopes

  def retrieve_scope(self, func_name, scopes):
    return [
        scopes[scope]
        for scope in scopes
        if scopes[scope].func_name == func_name
    ][0]

  def assertScopesEqual(self, src_scope, target_scope):
    self.assertEqual(src_scope.parent, target_scope.parent)
    self.assertEqual(src_scope.func_name, target_scope.func_name)
    self.assertScopesVarsEqual(src_scope, target_scope)

  def assertScopesVarsEqual(self, src_scope, target_scope):
    self.assertEqual(src_scope.locals, target_scope.locals)
    self.assertEqual(src_scope.free, target_scope.free)
    self.assertEqual(src_scope.nonlocals, target_scope.nonlocals)
    self.assertEqual(src_scope.globals, target_scope.globals)

  def test_locals(self):
    scope = scoping.Scope(None)
    scope.add_local('x')
    self.assertEqual(scope.locals, set('x'))

    scope.add_free('x')
    self.assertEqual(scope.locals,
                     set('x'))  # adding to free does NOT modify locals
    self.assertEqual(scope.free, set('x'))

  def test_free(self):
    scope = scoping.Scope(None)
    scope.add_free('x')
    self.assertEqual(scope.free, set('x'))

    scope.add_local('x')
    self.assertEqual(scope.locals, set('x'))
    self.assertEqual(scope.free, set())  # adding to locals DOES modify free

  def test_globals(self):
    scope = scoping.Scope(None)
    scope.add_global('x')
    self.assertEqual(scope.globals, set('x'))

    scope.add_local('y')
    self.assertEqual(scope.locals, set('y'))
    scope.add_global('y')
    self.assertEqual(scope.locals, set())
    self.assertEqual(scope.globals, set(['x', 'y']))

  def test_nonlocals(self):
    scope = scoping.Scope(None)
    scope.add_nonlocal('x')
    self.assertEqual(scope.nonlocals, set('x'))

    scope.add_local('y')
    self.assertEqual(scope.locals, set('y'))
    scope.add_nonlocal('y')
    self.assertEqual(scope.locals, set())
    self.assertEqual(scope.nonlocals, set(['x', 'y']))

  def test_build_locals(self):

    def func():
      a = 1
      b = a + 1
      return b

    scopes = self.get_scopes(func)
    self.assertLen(scopes, 1)
    func_scope = self.retrieve_scope('func', scopes)
    expected_scope = scoping.Scope(func_name='func')
    expected_scope.add_local('a')
    expected_scope.add_local('b')
    self.assertScopesEqual(func_scope, expected_scope)

  def test_locals_and_params(self):

    def func(x):
      y = 1
      return x + y

    scopes = self.get_scopes(func)
    self.assertLen(scopes, 1)
    func_scope = self.retrieve_scope('func', scopes)
    self.assertNotEqual(func_scope, None)
    expected_scope = scoping.Scope(func_name='func')
    expected_scope.add_local('x')
    expected_scope.add_local('y')
    self.assertScopesEqual(func_scope, expected_scope)

  def test_read_closure(self):

    def enclosing_func():
      x = 1

      def inner_func():
        return x

      return inner_func()

    scopes = self.get_scopes(enclosing_func)
    self.assertLen(scopes, 2)

    enclosing_scope = self.retrieve_scope('enclosing_func', scopes)
    self.assertNotEqual(enclosing_scope, None)

    inner_scope = self.retrieve_scope('inner_func', scopes)
    self.assertNotEqual(inner_scope, None)

    expected_enclosing = scoping.Scope(func_name='enclosing_func')
    expected_enclosing.add_local('x')
    expected_enclosing.add_local('inner_func')

    self.assertScopesEqual(enclosing_scope, expected_enclosing)

    expected_inner = scoping.Scope(enclosing_scope, func_name='inner_func')
    expected_inner.add_free('x')

    self.assertScopesEqual(inner_scope, expected_inner)

  def test_build_globals(self):

    def func():
      global z  # pylint: disable=global-variable-undefined
      z = 1

    scopes = self.get_scopes(func)
    self.assertLen(scopes, 1)

    scope = [
        scopes[scope] for scope in scopes if scopes[scope].func_name == 'func'
    ]

  def test_global_after(self):

    def global_after_vars(y):
      x = y  # pylint: disable=unused-variable,undefined-variable
      global x  # pylint: disable=global-variable-not-assigned

    scopes = self.get_scopes(global_after_vars)
    self.assertLen(scopes, 1)

    func_scope = self.retrieve_scope('global_after_vars', scopes)
    expected_func_scope = scoping.Scope(None, 'global_after_vars')
    expected_func_scope.add_global('x')
    expected_func_scope.add_local('y')

    self.assertScopesEqual(func_scope, expected_func_scope)

if __name__ == '__main__':
  test.main()

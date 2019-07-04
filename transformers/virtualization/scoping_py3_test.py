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
"""Tests for nonlocal variables using variables converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from pyctr.core import naming
from pyctr.core import parsing
from pyctr.sct import transformer
from pyctr.transformers.virtualization import scoping


class ScopingPy3Test(test.TestCase):

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

  def test_build_nonlocals(self):

    def outer():
      x = 1

      def inner():
        nonlocal x
        x = 2

      inner()

    scopes = self.get_scopes(outer)
    self.assertLen(scopes, 2)

    outer_scope = self.retrieve_scope('outer', scopes)
    expected_outer_scope = scoping.Scope(None, 'outer')
    expected_outer_scope.add_local('x')
    expected_outer_scope.add_local('inner')
    self.assertScopesEqual(outer_scope, expected_outer_scope)

    inner_scope = self.retrieve_scope('inner', scopes)
    expected_inner_scope = scoping.Scope(outer_scope, 'inner')
    expected_inner_scope.add_nonlocal('x')
    self.assertScopesEqual(inner_scope, expected_inner_scope)
    self.assertTrue(inner_scope.should_virtualize('x'))

  def test_build_nonlocals_no_virtualize(self):

    def outer():
      x = 1

      def inner():
        nonlocal x
        x = 2

      return inner

    inner = outer()
    scopes = self.get_scopes(inner)
    self.assertLen(scopes, 1)

    inner_scope = self.retrieve_scope('inner', scopes)
    self.assertFalse(inner_scope.should_virtualize('x'))


if __name__ == '__main__':
  test.main()

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
"""Contains code which builds function scopes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
from pyctr.sct import transformer


class Scope(object):
  """Object representing scope.

  Tracks variables visible within a scope.

  Attributes:
    parent: Scope, enclosing scope
    locals: set[Str], mutable variables visible in current scope
    free: set[Str], immutable variables visible in current scope
    nonlocals: set[Str], nonlocal variables
    globals: set[Str], global variables
    func_name: Str, name of associated function (for pretty printing)
  """

  def __init__(self, parent=None, func_name=None):
    self.parent = parent
    self.locals = set()
    self.free = set()
    self.nonlocals = set()
    self.globals = set()
    self.func_name = func_name

  def add_free(self, var):
    self.free.add(var)

  def add_local(self, var):
    if var in self.free:
      self.free.remove(var)
    self.locals.add(var)

  def add_nonlocal(self, var):
    if var in self.locals:
      self.locals.remove(var)
    self.nonlocals.add(var)

  def add_global(self, var):
    if var in self.locals:
      self.locals.remove(var)
    self.globals.add(var)

  def is_local(self, var):
    return var in self.locals

  def is_free(self, var):
    return var in self.free

  def is_global(self, var):
    return var in self.globals

  def is_nonlocal(self, var):
    return var in self.nonlocals

  def is_bound(self, var):
    return self.is_local(var) or self.is_global(var) or self.is_nonlocal(
        var) or (self.parent and self.parent.is_bound(var))

  def should_virtualize(self, var):
    if self.is_local(var):
      return True

    if self.is_nonlocal(var) or self.is_free(var):
      return self.parent and self.parent.should_virtualize(var)

    return False

  def __repr__(self):
    return ('Scope(parent={}, function={}, locals={}, free={}, nonlocals={}, '
            'globals={})').format(self.parent, self.func_name, self.locals,
                                  self.free, self.nonlocals, self.globals)


class ScopeTransformer(transformer.Base):
  """Tracks scope information to be used by VariableTransformer.

  Attributes:
    self.scopes: dict[gast.FunctionDef, Scope], map from FunctionDef to Scope
    self.scope: Scope, current scope
  """

  def __init__(self, ctx):
    super(ScopeTransformer, self).__init__(ctx.info)
    self.scopes = {}
    self.scope = None

  def _check_local(self, node, name):
    if not self.scope or self.scope.is_global(name) or self.scope.is_nonlocal(
        name):
      return

    if isinstance(node.ctx, gast.Store):
      self.scope.add_local(name)
    elif not self.scope.is_local(name):
      self.scope.add_free(name)

  def visit_FunctionDef(self, node):
    if self.scope:
      self.scope.add_local(node.name)

    scope = Scope(self.scope, node.name)
    self.scopes[id(node)] = scope
    self.scope = scope

    for arg in node.args.args:
      self.scope.add_local(arg.id)

    node = self.generic_visit(node)

    if self.scope.parent:
      self.scope.parent.add_local(node.name)

    self.scope = self.scope.parent
    return node

  def _visit_free(self, node, callback):
    node = self.generic_visit(node)

    for name in node.names:
      callback(name)

    return node

  def visit_Nonlocal(self, node):
    return self._visit_free(node, self.scope.add_nonlocal)

  def visit_Global(self, node):
    return self._visit_free(node, self.scope.add_global)

  def visit_For(self, node):
    if isinstance(node.target, gast.Tuple) or isinstance(
        node.target, gast.List):
      for target in node.target.elts:
        self._check_local(node.target, target.id)
    else:
      self._check_local(node.target, node.target.id)
    node.iter = self.generic_visit(node.iter)
    node.body = self.visit_block(node.body)
    node.orelse = self.visit_block(node.orelse)
    return node

  def visit_Name(self, node):
    node = self.generic_visit(node)
    self._check_local(node, node.id)
    return node

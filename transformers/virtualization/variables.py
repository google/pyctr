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
"""Handles virtualization of reads/writes for variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
from pyctr.sct import templates
from pyctr.sct import transformer
from pyctr.transformers.virtualization import scoping


class VariableTransformer(transformer.Base):
  """Virtualizes reads/writes of variables.

  Attributes:
    ctx: transformer.EntityContext, see transformer.Base
    overload: overloads.Overload, the overload module and gen_sym
    scopes: dict[gast.FunctionDef, scoping.Scope], map from FunctionDef to Scope
      (see scoping.ScopeTransformer)
    scope: scoping.Scope, the current Scope
  """

  def __init__(self, ctx, overload, scopes):
    super(VariableTransformer, self).__init__(ctx.info)
    self.ctx = ctx
    self.overload = overload
    self.scopes = scopes
    self.scope = None

  def visit_FunctionDef(self, node):
    assert id(node) in self.scopes

    self.scope = self.scopes[id(node)]

    arg_names = [arg.id for arg in node.args.args]
    n_arg_names = [
        self.ctx.namer.new_symbol(arg, set(arg_names)) for arg in arg_names
    ]

    init_nodes = []

    for var in self.scope.locals:
      init_template = 'lhs = overload.init(lhs_name)'
      init_node = templates.replace(
          init_template,
          lhs=var,
          lhs_name='"{}"'.format(var),
          overload=self.overload.symbol_name)
      init_nodes.extend(init_node)

    arg_nodes = []

    for (arg, n_arg) in zip(arg_names, n_arg_names):
      arg_template = 'overload.assign(lhs, rhs)'
      arg_node = templates.replace(
          arg_template, lhs=arg, overload=self.overload.symbol_name, rhs=n_arg)
      arg_nodes.extend(arg_node)

    node.body = self.visit_block(node.body)

    if self.scope.parent and self.scope.parent.is_local(node.name):
      template = """
        def new_fun_name(args):
          inits
          arg_nodes
          body
        overload.assign(fun_name, new_fun_name)
      """

      node = templates.replace(
          template,
          new_fun_name=self.ctx.namer.new_symbol(node.name, set([node.name])),
          args=n_arg_names,
          arg_nodes=arg_nodes,
          inits=init_nodes,
          body=node.body,
          overload=self.overload.symbol_name,
          fun_name=node.name,
      )
    else:
      template = """
        def fun_name(args):
          inits
          arg_nodes
          body
      """

      node = templates.replace(
          template,
          fun_name=node.name,
          args=n_arg_names,
          arg_nodes=arg_nodes,
          inits=init_nodes,
          body=node.body,
      )

    self.scope = self.scope.parent
    return node

  def visit_Assign(self, node):
    # TODO(b/123943188): Handle multiple assignment
    node.value = self.visit(node.value)

    lhs = node.targets[0].id
    rhs = node.value

    if not self.scope.should_virtualize(lhs):
      return node

    node = templates.replace(
        'overload.assign(lhs, rhs)',
        lhs=lhs,
        rhs=rhs,
        overload=self.overload.symbol_name)

    return node

  def visit_AugAssign(self, node):
    # TODO(b/123943188): Implement AugAssign
    raise NotImplementedError('AugAssign not yet implemented.')

  def _make_target_assign(self, target, n_target, i, overload):
    return templates.replace(
        'overload.assign(target, n_target[{}])'.format(i),
        target=target,
        n_target=n_target,
        overload=self.overload.symbol_name)

  def visit_For(self, node):
    node.iter = self.visit(node.iter)
    node.body = self.visit_block(node.body)
    node.orelse = self.visit_block(node.orelse)

    targets = []

    if isinstance(node.target, gast.Tuple) or isinstance(
        node.target, gast.List):
      for target in node.target.elts:
        targets.append(target)
    elif isinstance(node.target, gast.Name):
      targets.append(node.target)
    else:
      raise ValueError(
          'For target must be gast.Tuple, gast.List, or gast.Name, got {}.'
          .format(type(node.target)))

    n_target = self.ctx.namer.new_symbol('n_target',
                                         set([target.id for target in targets]))
    target_assigns = []

    if len(targets) > 1:
      for i, target in enumerate(targets):
        target_assign = self._make_target_assign(target, n_target, i,
                                                 self.overload)
        target_assigns.extend(target_assign)
    else:
      target_assign = templates.replace(
          'overload.assign(target, n_target)',
          overload=self.overload.symbol_name,
          target=targets[0],
          n_target=n_target)
      target_assigns.extend(target_assign)

    template = """
      for n_target in iter:
        target_assigns
        body
      else:
        orelse
    """

    node = templates.replace(
        template,
        n_target=n_target,
        iter=node.iter,
        target_assigns=target_assigns,
        body=node.body,
        orelse=node.orelse)

    return node

  def visit_Name(self, node):
    node = self.generic_visit(node)

    if not hasattr(self.overload.module, 'read'):
      return node

    if self.scope.should_virtualize(node.id):
      node = templates.replace_as_expression(
          'overload.read(id)', overload=self.overload.symbol_name, id=node.id)
    return node


def transform(node, ctx, overload):
  sc = scoping.ScopeTransformer(ctx)
  node = sc.visit(node)
  node = VariableTransformer(ctx, overload, sc.scopes).visit(node)
  return node

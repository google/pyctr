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
"""Handles control flow statements: while, for, if."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
from pyctr.analysis import activity
from pyctr.core import anno
from pyctr.core import qual_names
from pyctr.sct import templates
from pyctr.sct import transformer


class ControlFlowTransformer(transformer.Base):
  """Transforms control flow structures like loops and conditionals."""

  def __init__(self, ctx, overload):
    super(ControlFlowTransformer, self).__init__(ctx.info)
    self.ctx = ctx
    self.overload = overload

  def visit_If(self, node):
    body_scope = anno.getanno(node, anno.Static.BODY_SCOPE)
    orelse_scope = anno.getanno(node, anno.Static.ORELSE_SCOPE)
    modified_in_cond = body_scope.modified | orelse_scope.modified

    node = self.generic_visit(node)

    if not hasattr(self.overload.module, 'if_stmt'):
      return node

    template = """
      def test_name():
        return test
      def body_name():
        body
      def orelse_name():
        orelse
      overload.if_stmt(test_name, body_name, orelse_name, (local_writes,))
    """

    node = templates.replace(
        template,
        overload=self.overload.symbol_name,
        test_name=self.ctx.namer.new_symbol('if_test', set()),
        test=node.test,
        body_name=self.ctx.namer.new_symbol('if_body', set()),
        body=node.body,
        orelse_name=self.ctx.namer.new_symbol('if_orelse', set()),
        orelse=node.orelse if node.orelse else gast.Pass(),
        local_writes=tuple(modified_in_cond))

    return node

  def visit_While(self, node):
    body_scope = anno.getanno(node, anno.Static.BODY_SCOPE)
    orelse_scope = anno.getanno(node, anno.Static.ORELSE_SCOPE)
    modified_in_cond = body_scope.modified | orelse_scope.modified

    node = self.generic_visit(node)

    if not hasattr(self.overload.module, 'while_stmt'):
      return node

    template = """
      def test_name():
        return test
      def body_name():
        body
      def orelse_name():
        orelse
      overload.while_stmt(test_name, body_name, orelse_name, (local_writes,))
    """

    node = templates.replace(
        template,
        overload=self.overload.symbol_name,
        test_name=self.ctx.namer.new_symbol('while_test', set()),
        test=node.test,
        body_name=self.ctx.namer.new_symbol('while_body', set()),
        body=node.body,
        orelse_name=self.ctx.namer.new_symbol('while_orelse', set()),
        orelse=node.orelse if node.orelse else gast.Pass(),
        local_writes=tuple(modified_in_cond))

    return node

  def _make_target_init(self, target, overload):
    return templates.replace(
        'target = overload.init(target_name)',
        target=target,
        target_name='"{}"'.format(target.id),
        overload=self.overload.symbol_name)

  def visit_For(self, node):
    body_scope = anno.getanno(node, anno.Static.BODY_SCOPE)
    orelse_scope = anno.getanno(node, anno.Static.ORELSE_SCOPE)
    modified_in_cond = body_scope.modified | orelse_scope.modified

    node = self.generic_visit(node)

    if not hasattr(self.overload.module, 'for_stmt'):
      return node

    # TODO(jmd1011): Handle extra_test

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

    target_inits = [
        self._make_target_init(target, self.overload) for target in targets
    ]

    template = """
      target_inits
      def body_name():
        body
      def orelse_name():
        orelse
      overload.for_stmt(target, iter_, body_name, orelse_name, (local_writes,))
    """

    node = templates.replace(
        template,
        target_inits=target_inits,
        target=node.target,
        body_name=self.ctx.namer.new_symbol('for_body', set()),
        body=node.body,
        orelse_name=self.ctx.namer.new_symbol('for_orelse', set()),
        orelse=node.orelse if node.orelse else gast.Pass(),
        overload=self.overload.symbol_name,
        iter_=node.iter,
        local_writes=tuple(modified_in_cond))

    return node


def transform(node, ctx, overload):
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, parent_scope=None, overload=overload)
  node = ControlFlowTransformer(ctx, overload).visit(node)
  return node

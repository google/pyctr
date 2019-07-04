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
"""Handles logical ops: and, or, not."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
from pyctr.sct import templates
from pyctr.sct import transformer


class LogicalOpTransformer(transformer.Base):
  """Transforms logical ops and, or, not."""

  def __init__(self, ctx, overload):
    super(LogicalOpTransformer, self).__init__(ctx.info)
    self.ctx = ctx
    self.overload = overload

  def _make_lambda_nodes(self, lst):
    lambda_nodes = []

    for y in lst:
      lambda_node = templates.replace_as_expression('lambda: y', y=y)
      lambda_nodes.append(lambda_node)

    return lambda_nodes

  def _handle_boolop(self, node, func):
    assert isinstance(node, gast.BoolOp)

    node = self.generic_visit(node)
    lambda_nodes = self._make_lambda_nodes(node.values[1:])
    node = templates.replace_as_expression(
        'overload.func(x, (operands,))',
        func=func,
        overload=self.overload.symbol_name,
        x=node.values[0],
        operands=lambda_nodes)

    return node

  def visit_BoolOp(self, node):
    if isinstance(node.op, gast.And) and hasattr(self.overload.module, 'and_'):
      return self._handle_boolop(node, 'and_')
    elif isinstance(node.op, gast.Or) and hasattr(self.overload.module, 'or_'):
      return self._handle_boolop(node, 'or_')

    node = self.generic_visit(node)
    return node

  def _overload_Not(self, node):
    assert isinstance(node, gast.UnaryOp)

    node = self.generic_visit(node)
    node = templates.replace_as_expression(
        'overload.not_(x)',
        overload=self.overload.symbol_name,
        x=node.operand)

    return node

  def visit_UnaryOp(self, node):
    if isinstance(node.op, gast.Not) and hasattr(self.overload.module, 'not_'):
      return self._overload_Not(node)

    node = self.generic_visit(node)
    return node


def transform(node, ctx, overload):
  node = LogicalOpTransformer(ctx, overload).visit(node)
  return node

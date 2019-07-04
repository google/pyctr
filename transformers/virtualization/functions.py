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
"""Handles virtualization of function calls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
from pyctr.core import ast_util
from pyctr.sct import templates
from pyctr.sct import transformer


class FunctionCallTransformer(transformer.Base):
  """Virtualizes function calls.

  Attributes:
    ctx: transformer.EntityContext, see transformer.Base
    overload: overloads.Overload, the overload module
  """

  def __init__(self, ctx, overload):
    super(FunctionCallTransformer, self).__init__(ctx.info)
    self.overload = overload

  def is_overload_call(self, node):
    """True if the node is a call to a function on the overload module."""
    if isinstance(node.func, gast.Attribute):
      if isinstance(node.func.value, gast.Name):
        if node.func.value.id == self.overload.symbol_name:
          return True
    return False

  def visit_Call(self, node):
    node = self.generic_visit(node)

    if not hasattr(self.overload.module, 'call'):
      return node

    if self.is_overload_call(node):
      return node

    starred_arg = None
    normal_args = []
    for a in node.args:
      if isinstance(a, gast.Starred):
        assert starred_arg is None, 'Multiple *args should be impossible.'
        starred_arg = a
      else:
        normal_args.append(a)
    if starred_arg is None:
      args = templates.replace_as_expression('(args,)', args=normal_args)
    else:
      args = templates.replace_as_expression(
          '(args,) + tuple(stararg)',
          stararg=starred_arg.value,
          args=normal_args)

    kwargs_arg = None
    normal_keywords = []
    for k in node.keywords:
      if k.arg is None:
        assert kwargs_arg is None, 'Multiple **kwargs should be impossible.'
        kwargs_arg = k
      else:
        normal_keywords.append(k)
    if kwargs_arg is None:
      kwargs = ast_util.keywords_to_dict(normal_keywords)
    else:
      kwargs = templates.replace_as_expression(
          'dict(kwargs, **keywords)',
          kwargs=kwargs_arg.value,
          keywords=ast_util.keywords_to_dict(normal_keywords))

    template = """
      overload.call(func, args, kwargs)
    """
    node = templates.replace_as_expression(
        template,
        overload=self.overload.symbol_name,
        func=node.func,
        args=args,
        kwargs=kwargs)

    return node


def transform(node, ctx, overload):
  return FunctionCallTransformer(ctx, overload).visit(node)

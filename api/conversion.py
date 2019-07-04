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
"""Internal API for Pyct."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

from pyctr.api import config
from pyctr.core import naming
from pyctr.core import parsing
from pyctr.sct import templates
from pyctr.sct import transformer
import six


def _transform(source, ctx, overload, transformers):
  for tr in transformers:
    source = apply_(source, ctx, tr, overload)
  return source


def _wrap_in_generator(func, source, namer, overload):
  """Wraps the source code in a generated function.

  Args:
    func: the original function
    source: the generated source code
    namer: naming.Namer, used for naming vars
    overload: config.VirtualizationConfig

  Returns:
    The generated function with a new closure variable.
  """

  nonlocals = []

  for var in six.get_function_code(func).co_freevars:
    # We must generate dummy vars so the generated function has the same closure
    # as the original function.
    free_template = 'var = None'
    nonlocal_node = templates.replace(free_template, var=var)
    nonlocals.extend(nonlocal_node)

  gen_fun_name = namer.new_symbol('gen_fun', set())
  template = """
    def gen_fun(overload):
      nonlocals

      program

      return f_name
  """

  ret = templates.replace(
      template,
      gen_fun=gen_fun_name,
      nonlocals=nonlocals,
      overload=overload.symbol_name,
      program=source,
      f_name=func.__name__)

  converted_module, _ = parsing.ast_to_object(ret)
  outer_func = getattr(converted_module, gen_fun_name)
  return outer_func(overload.module)


def _attach_closure(original_func, gen_func):
  """Attaches original_func's closure to gen_func.

  Args:
    original_func: original function
    gen_func: generated function

  Returns:
    A new function with the complete closure
  """

  closure = ()

  gen_code = six.get_function_code(gen_func)
  gen_closure = six.get_function_closure(gen_func)

  if not gen_closure:
    gen_closure = ()

  original_code = six.get_function_code(original_func)
  original_closure = six.get_function_closure(original_func)

  if not original_closure:
    original_closure = ()

  gen_dict = {
      free_var: cell
      for free_var, cell in zip(gen_code.co_freevars, gen_closure)
  }

  original_dict = {
      free_var: cell
      for free_var, cell in zip(original_code.co_freevars, original_closure)
  }

  gen_dict.update(original_dict)

  closure = tuple([gen_dict[cell] for cell in gen_code.co_freevars])

  return types.FunctionType(
      gen_code,
      original_func.__globals__,
      argdefs=original_func.__defaults__,
      closure=closure)


def convert(func, overload_module, transformers):
  """Main entry point for converting a function using Pyct.

  Args:
    func: function to be converted
    overload_module: module containing overloaded functionality
    transformers: list of transformers to be applied

  Returns:
    gen_func: converted function
  """
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
  overload_name = ctx.namer.new_symbol('overload', set())
  overload = config.VirtualizationConfig(overload_module, overload_name)

  source = _transform(source, ctx, overload, transformers)
  gen_func = _wrap_in_generator(func, source, namer, overload)
  gen_func = _attach_closure(func, gen_func)
  return gen_func


def apply_(node, ctx, transformer_module, overload):
  node = transformer_module.transform(node, ctx, overload)
  return node

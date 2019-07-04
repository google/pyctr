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
"""Converting code to AST.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import imp
import inspect
import os
import tempfile
import textwrap

import astor
import gast
import six


def parse_entity(entity):
  """Returns the AST of given entity."""
  source = inspect.getsource(entity)

  def fail(comment):
    raise ValueError(
        'Failed to parse source code of {}, which Python reported as:\n{}\n'
        '{}'.format(entity, source, comment))

  # Comments and multiline strings can appear at arbitrary indentation levels,
  # causing textwrap.dedent to not correctly dedent source code.
  # TODO(b/115884650): Automatic handling of comments/multiline strings.
  source = textwrap.dedent(source)

  try:
    return parse_str(source), source

  except IndentationError:
    # The text below lists the causes of this error known to us. There may
    # be more.
    fail('This may be caused by multiline strings or comments not indented at'
         'the same level as the code.')

  except SyntaxError as e:
    if not inspect.isfunction(entity) or entity.__name__ != '<lambda>':
      raise

    # Certain entities, like lambdas, only hold the raw code lines which defined
    # them, which may include surrounding tokens and may be syntactically
    # invalid out of context. For example:
    #
    #     l = (
    #         lambda x: x,)[0]
    #
    # will have the dedented source "lambda x: x,)[0]"
    # Here we make an attempt to stip away the garbage by looking at the
    # information in the syntax error.
    lines = source.split('\n')
    lineno, offset = e.lineno, e.offset  # 1-based

    # Give up if there's nothing we can chip away.
    if len(lines) == lineno and len(lines[-1]) == offset:
      fail('If this is a lambda function, the error may be avoided by creating'
           ' the lambda in a standalone statement.')

    # Drop all lines following the error location
    # TODO(mdanatg): What's with the pylint errors?
    lines = lines[:lineno]  # pylint:disable=invalid-slice-index
    # Drop all characters following the error location
    lines[-1] = lines[-1][:offset - 1]  # pylint:disable=invalid-slice-index
    new_source = '\n'.join(lines)

    try:
      return parse_str(new_source), new_source
    except SyntaxError as e:
      fail('If this is a lambda function, the error may be avoided by creating'
           ' the lambda in a standalone statement. Tried to strip down the'
           ' source to:\n{}\nBut that did not work.'.format(new_source))


def parse_str(src):
  """Returns the AST of given piece of code."""
  # TODO(mdanatg): This should exclude the module things are autowrapped in.

  if six.PY2 and '.print(' in src:
    # This special treatment is required because gast.parse is not aware of
    # whether print_function was present in the original context.
    src = 'from __future__ import print_function\n' + src
    parsed_module = gast.parse(src)
    parsed_module.body = parsed_module.body[1:]
  else:
    parsed_module = gast.parse(src)

  return parsed_module


def parse_expression(src):
  """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression

  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
  node = parse_str(src)
  assert isinstance(node, gast.Module)
  if len(node.body) != 1 or not isinstance(node.body[0], gast.Expr):
    raise ValueError(
        'Expected a single expression, found instead %s' % node.body)
  return node.body[0].value


def ast_to_source(node, indentation='  '):
  """Return the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: The string to use for indentation.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and Pyct generated code.
  """
  if not isinstance(node, (list, tuple)):
    node = (node,)
  generator = astor.code_gen.SourceGenerator(indentation, False,
                                            astor.string_repr.pretty_string)

  for n in node:
    if isinstance(n, gast.AST):
      n = gast.gast_to_ast(n)
    generator.visit(n)
    generator.result.append('\n')

  # In some versions of Python, literals may appear as actual values. This
  # ensures everything is string.
  code = ''.join(map(str, generator.result))

  # Strip leading blank lines.
  code_lines = code.split('\n')
  trimmed_code_lines = []
  for l in code_lines:
    if l.rstrip() or trimmed_code_lines:
      trimmed_code_lines.append(l)
  code = '\n'.join(trimmed_code_lines)

  # Work around the reference cycle generated by astor.
  # See https://github.com/berkerpeksag/astor/blob/55dd323f7d8d696610c703c0296763c567685c31/astor/code_gen.py#L162  # pylint:disable=line-too-long
  # Reference cycles are quite disliked by TensorFlow's tests.
  if hasattr(generator, 'write'):
    generator.write = None
  del generator

  return code


def ast_to_object(nodes,
                  indentation='  ',
                  include_source_map=False,
                  source_prefix=None,
                  delete_on_exit=True):
  """Return the Python objects represented by given AST.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
      object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether to attach a source map to the compiled
      object. Also see origin_info.py.
    source_prefix: Optional[Text], string to print as-is into the source file.
    delete_on_exit: bool, whether to delete the temporary file used for
      compilation on exit.

  Returns:
    compiled_nodes: A module object containing the compiled source code.
    source: The source code of the compiled object
  Raises:
    ValueError: If ag_source_map__ is already in the namespace of the compiled
    nodes.
  """
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  source = ast_to_source(nodes, indentation=indentation)

  if source_prefix:
    source = source_prefix + '\n' + source

  with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    module_name = os.path.basename(f.name[:-3])
    f.write(source)

    if isinstance(nodes, (list, tuple)):
      indices = range(-len(nodes), 0)
    else:
      indices = (-1,)

    if include_source_map:
      # TODO(mdanatg): Break this dependency cycle.
      from pyctr.core import origin_info  # pylint:disable=g-import-not-at-top
      source_map = origin_info.create_source_map(nodes, source, f.name, indices)

  # TODO(mdanatg): Try flush() and delete=False instead.
  if delete_on_exit:
    atexit.register(lambda: os.remove(f.name))
  compiled_nodes = imp.load_source(module_name, f.name)

  # TODO(znado): Clean this up so we don't need to attach it to the namespace.
  # We cannot get the rewritten function name until it is too late so templating
  # is hard, and this cleanly fixes the issues encountered with nested functions
  # because this is attached to the outermost one.
  if include_source_map:
    # TODO(mdanatg): This name should be decided by the caller.
    source_map_name = 'ag_source_map__'
    # TODO(jmd1011): This is AutoGraph specific -- needs to be made generic.
    assert source_map_name not in compiled_nodes.__dict__, (
        'cannot convert %s because it has namespace attribute "%s", which is '
        'reserved for AutoGraph.') % (compiled_nodes, source_map_name)
    compiled_nodes.__dict__[source_map_name] = source_map

  return compiled_nodes, source

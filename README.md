IMPORTANT: Pyctr is prototype software, and under active development. Expect rough
edges and bugs, but if you try it, we appreciate early feedback! We'd also love
contributions ([please see our contributing guidelines](CONTRIBUTING.md).

# What is Pyctr?

Pyctr is a virtualization and source code transformation engine for Python. It
enables multi-stage
programming in Python, and is the core functionality of AutoGraph extracted into
a standalone module (i.e., no dependencies on TensorFlow).

Language virtualization (as defined by
[Chafi et. al.](https://dl.acm.org/citation.cfm?id=1869527))
is a methodology in which language constructs are overloadable as virtual
methods. Virtualization enables framework
developers to add their own overloads for Python constructs which are not
normally overloadable. In this way Pyctr can be seen as an extension of Python's
built in operator overloading support.
Python allows one to provide an implementation of `__add__`
to override the meaning of the `+` symbol. In an analogous way, Pyctr
essentially allows developers to define their own `__if__`
function that will be executed in place of Pythons `if` statement.

Pyctr aims to virtualize all of Python, allowing DSL and framework developers to
overload Python syntax with custom behavior. Pyctr currently supports many
Python features including, but not limited to,
control-flow statements (e.g. `if`, `while`, and `for`), function calls, and
variable access. We're continually working on support for additional Python
constructs.

# Getting started with Pyctr

## Background

Pyctr is a Python code transformation library currently distributed as a part of
TensorFlow AutoGraph. AutoGraph is a tool used to convert eager-style Python
code, including control-flow, into code that generates TensorFlow graphs. Pyctr
inside AutoGraph is a refinement of the code transformation tools generated as
part of the Tangent automatic differentiation project.

This repository contains a version of Pyctr that does not depend on TensorFlow
and can be used
to provide AutoGraph-like functionality (reusing Python syntax for
framework-specific functionality) for any framework that can be called from
Python.

This document describes how to get started using Pyctr (also see the API
documentation). There are two use-cases to consider: users of Pyctr, and users
of systems which use Pyctr. In this document, we refer to users directly using
Pyctr as “DSL developers;” they are using Pyctr to repurpose Python as a DSL,
and simply providing the domain-specific logic. We refer to the other category
of users as “end users.”

## Examples

If you want to skip ahead to looking at some examples, Pyctr provides some in
the `examples` module.

## DSL Developers

### Designing a Semantic Overload

In using Pyctr, one must first determine which features of Python require
virtualization. For example, a DSL developer may only require function calls to
be virtualized, in other cases (such as when translating an entire program) it
may be necessary to virtualize everything. Currently, these developers must find
which transformer contains the necessary constructs and supply that as an
argument for Pyctr’s conversion method; soon this will be inferred automatically
by inspecting a property of the supplied overloads object. Pyctr currently makes
the following transformers available:

* variables: for variable virtualization
* control_flow: for control flow structures such as for/while loops and if
  statements (Note: control_flow requires that the variables transformer has
  taken place. Failure to perform these in the correct order may yield
  unintended consequences.)
* functions: for function call virtualization
* logical_ops: for and, or, not virtualization

We provide a detailed explanation of the API exposed by each transformer below.

#### `variables`
Virtualized methods exposed:

* init(name): this occurs before the first assignment of a variable (typically
  immediately before, but there are some exceptions). This will appear in the
  pattern x = overload.init(‘x’)
* assign: this represents variable assignment, and replaces the pattern
  lhs = rhs with overload.assign(lhs, rhs)
* read: this represents a variable read, and replaces the pattern x with
  overload.read(x)

Virtualizing variables requires a separate pass of the AST to construct the
proper scopes. Pyctr currently handles all global and nonlocal scopes as
expected, but does not handle classes (yet!). The code for this may be found in
transformers/virtualization/scoping.py.

For more details, see transformers/virtualizations/variables.py.
#### `control_flow`
Virtualized methods exposed:

* if_stmt(cond, body, orelse, local_writes): this replaces idiomatic if
  statements with functions representing the condition, then branch, and else
  branch, and a call to overload.if_stmt
* while_stmt(cond, body, orelse, local_writes): this replaces idiomatic while
  loops with functions representing the condition, loop body, and orelse body,
  and a call to overload.while_stmt
* for_stmt(target, iter_, body, orelse, local_writes): this replaces idiomatic
  for loops with functions representing the loop body and orelse body, and a
  call to overload.for_stmt

For more details, see transformers/virtualizations/control_flow.py.

#### `functions`
Virtualized methods exposed:

* call(func, args, kwargs): this replaces all function calls (other than those
  whitelisted) with a call to overload.call

For more details, see transformers/virtualizations/functions.py.

#### `logical_ops`
Virtualized methods exposed:

* and_(a, b): replaces pattern a and b with overload.and_(a, b)
* or_(a, b): replaces pattern a or b with overload.or_(a, b)
* not_(x): replaces pattern not x with overload.not_(x)

Note that and_ and or_ expect a variable number of parameters for b in order to
match Python’s semantics (i.e., allowing chaining).

For more details, see transformers/virtualizations/logical_ops.py.

### Constructing an Overload
Examples of building overloads objects may be found in the `examples` module.

A simplified view is that a DSL developer creates an object upon which the
necessary attributes are defined. For example, one may create a module which
converts logical operations to Z3Py operations (see `examples/z3py/z3py.py`):

```
def and_(a, b):
  return z3.And(a, b)
```

```
def or_(a, b):
  return z3.Or(a, b)
```

```
def not_(x):
  return z3.Not(x)
```

The following code defines one of De Morgan’s laws using idiomatic Python
logical operators and converts it to Z3Py:

```
def demorgan(a, b):
  return (a and b) == (not (not a or not b))
```

```
converted_demorgan = pyctr.convert(demorgan, z3py, transformers=[logical_ops])
```

Note that `z3py` here represents the module we defined above.

### Exposing a Decorator
In order to provide an easy entrypoint for users, DSL developers may choose to
construct a decorator which may be applied to a function for conversion. This
may look as follows:

```
def pytorch_to_tf(func):
  return pyctr.convert(func, my_overload_object, transformers=my_transformers)
```

## End Users
If a DSL developer has chosen to expose a decorator as the entrypoint for Pyctr
conversion (as shown above), end users may simply do the following:

```
@pytorch_to_tf
def my_function(x, y):
  return torch.rand(x) * torch.rand(y)
```

Alternatively, end users may use the `Pyctr.convert` method directly.

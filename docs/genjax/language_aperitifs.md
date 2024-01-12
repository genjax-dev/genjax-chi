# Language apéritifs

> This page assumes that the reader has familiarity with trace-based probabilistic programming systems.

The implementation of GenJAX adhers to commonly accepted JAX idioms (1) and modern functional programming patterns (2).
{ .annotate }

1.  One example: _everything_ is a [Pytree](https://github.com/patrick-kidger/equinox). Implies another: everything is JAX traceable by default.
2.  _Modern_ here meaning patterns concerning the composition of effectful computations via effect handling abstractions.

GenJAX consists of a **set of languages** based around transforming pure functions to apply semantic transformations. On this page, we'll provide a taste of some of these languages.

## The builtin language

GenJAX provides a builtin language which supports a `trace` primitive and the ability to invoke other generative functions as callees:

```python
@genjax.static
def submodel():
  x = trace("x", normal)(0.0, 1.0) # explicit
  return x

@genjax.static
def model():
  x = submodel() @ "sub" # sugared
  return x
```

The `trace` call is a JAX primitive which is given semantics by transformations which implement the semantics of inference interfaces described in [Generative functions]().

Addresses (here, `"x"` and `"sub"`) are important - addressed random choices within `trace` allow us to structure the address hierarchy for the _measure over choice maps_ which generative functions in this language define.

Because convenient idioms for working with addresses is so important in Gen, the generative functions from the builtin language also support a form of "splatting" addresses into a caller.

```python
@genjax.static
def model():
  x = submodel.inline()
  return x
```

Invoking the `submodel` via the `inline` interface here means that the addresses in `submodel` are flattened into the address level for the `model`. If there's overlap, that's a problem! But GenJAX will yell at you for that.

## Structured control flow with combinators

The base modeling language is the `BuiltinGenerativeFunction` language shown above. The builtin language is based on pure functions, with the interface semantics implemented using program transformations. But we'd also like to take advantage of structured control flow in our generative computations.

Users gain access to structured control flow via _combinators_, other generative function mini-languages which implement the interfaces in control flow compatible ways.

```python
@functools.partial(genjax.Map, in_axes=(0, 0))
@genjax.static
def kernel(x, y):
  z = normal(x + y, 1.0) @ "z"
  return z
```

This defines a `MapCombinator` generative function - a generative function whose interfaces take care of applying `vmap` in the appropriate ways (1).
{ .annotate }

1.  Read: compatible with JIT, gradients, and incremental computation.

`MapCombinator` has a vectorial friend named `UnfoldCombinator` which implements a `scan`-like pattern of generative computation.


```python
@functools.partial(genjax.Unfold, max_length = 10)
@genjax.static
def scanner(prev, static_args):
  sigma, = static_args
  new = normal(prev, sigma) @ "z"
  return new
```

`UnfoldCombinator` allows the expression of general state space models - modeled as a generative function which supports a dependent-for (1) control flow pattern.
{ .annotate }

1. Dependent-for means that each iteration may depend on the output from the previous iteration. Think of `jax.lax.scan` here.

`UnfoldCombinator` allows uncertainty over the length of the chain:

```python
@genjax.static
def top_model(p):
  length = truncated_geometric(10, p) @ "l"
  initial_state = normal(0.0, 1.0) @ "init"
  sigma = normal(0.0, 1.0) @ "sigma"
  (v, xs) = scanner(length, initial_state, sigma)
  return v
```

Here, `length` is drawn from a truncated geometric distribution, and determines the index range of the chain which participates in the generative computation.

Of course, combinators are composable.

```python
@functools.partial(genjax.Map, in_axes = (0, ))
@genjax.static
def top_model(p):
  length = truncated_geometric(10, p) @ "l"
  initial_state = normal(0.0, 1.0) @ "init"
  sigma = normal(0.0, 1.0) @ "sigma"
  (v, xs) = scanner(length, initial_state, sigma)
  return v
```

Now we're describing a broadcastable generative function whose internal choices include a chain-like generative structure with dynamic truncation using padding. And we could go on!

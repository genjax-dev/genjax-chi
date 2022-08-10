# genjax

> [**Gen**](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [**JAX**](https://github.com/google/jax)

A concise encoding of Gen using zero-cost effect handling/tracing built on top of `jax`.

- Presents a modeling language based on the space of pure Python functions acceptable by `jax`: models are pure functions from `(PRNGKey, *args)` to `(PRNGKey, retval)`.
- Exposes [the generative function interface](https://www.gen.dev/stable/ref/gfi/) as staged effect handlers built on top of `jax`. (Roughly -- see documentation for exact signatures/return types):
  - `Simulate` (sample from normalized measure)
  - `Generate` (condition the generative function, and importance sample with model as prior)
  - `ArgumentGradients` (compute gradient of `logpdf` with respect to arguments)
  - `ChoiceGradients` (compute gradient of `logpdf` with respect to values of random choices)
- Should support usage of any computations acceptable by JAX (tbd) within generative function programs.

> **Early stage** expect 🔪 sharp edges 🔪

## Example

```python
import jax
import genjax as gex

# A `genjax` generative function is a pure Python function from
# `(PRNGKey, *args)` to `(PRNGKey, retval)`
#
# The programmer is free to use other JAX primitives, etc -- as desired.
#
# The models below are rather simplistic, but demonstrate
# proof of concept.
def g(key, x):
    key, m1 = gex.trace("m0", gex.Bernoulli)(key, x)
    return (key, m1)


# @gex(x = ShapedArray(shape=(2,), dtype=float))
def f(key, x):
    key, m0 = gex.trace("m0", gex.Bernoulli)(key, x)
    key, m1 = gex.trace("m1", gex.Normal)(key)
    key, m2 = gex.trace("m2", gex.Normal)(key)
    key, m3 = gex.trace("m3", gex.Bernoulli)(key, m1)
    key, m4 = gex.trace("m4", g)(key, m1)
    return key, (2 * m1 * m2, m0, m3, m4)


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

# This just shows our raw (not yet desugared/codegen) syntax.
expr = gex.lift(f, key, 0.3)
print(expr)

# Here's how you access the `simulate` GFI.
tr = jax.jit(gex.simulate(f))(key, 0.3)

# Here's how you access the `generate` GFI.
chm = {("m1",): 0.3, ("m2",): 0.5}
w, tr = jax.jit(gex.generate(f, chm))(key, 0.3)

# Here's how you access the `arg_grad` interface.
arg_grad = jax.jit(gex.arg_grad(f, tr, [1]))(key, 0.3)
print(arg_grad)

# Here's how you access the `choice_grad` interface.
fn = jax.jit(gex.choice_grad(f, tr, key, 0.3))
chm = {("m1",): 0.3, ("m2",): 0.5}
choice_grad, choices = fn(chm)
print(choice_grad)
chm = {("m1",): 0.3, ("m2",): 1.0}
choice_grad, choices = fn(chm)
print(choice_grad)
```

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched) by the compositional tracing provided by `jax`.

This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - sans automatic insertion of staging annotations. The inference interfaces exposed handle lifting/splicing/jitting internally (manually).

## Tour

[Jump into the tour!](/tour.py)

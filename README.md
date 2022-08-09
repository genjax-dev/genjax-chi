# gex

> [**Ge**n](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [JA**X**](https://github.com/google/jax)

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
import gex

# A `gex` generative function is a pure Python function from
# `(PRNGKey, *args)` to `(PRNGKey, retval)`
#
# The programmer is free to use other JAX primitives, etc -- as desired.
def g(key, x):
    key, m1 = gex.trace("m1", gex.Bernoulli)(key, x)
    return (key, m1)

def f(key, x):
    key, m1 = gex.trace("m1", gex.Bernoulli)(key, x)
    key, m2 = gex.trace("m2", gex.Bernoulli)(key, x)
    key, m3 = gex.trace("m3", g)(key, x)  # We support hierarchical models.
    return (key, 2 * (m1 + m2 + m3))

# Initialize a PRNG key.
key = jax.random.PRNGKey(314159)

# This just shows our raw (not yet desugared/codegen) syntax.
expr = gex.lift(f, key, 0.3)
print(expr)

# Here's how you access the `simulate` GFI.
fn = gex.Simulate().jit(f)(key, 0.3)
tr = fn(key, 0.3)
print(tr.get_choices())

# Here's how you access the `generate` GFI.
chm = {("m1",): True}
fn = gex.Generate(chm).jit(f)(key, 0.3)
w, tr = fn(key, 0.3)
print((w, tr.get_choices()))

# Here's how you access argument gradients --
# the second argument to `gex.ArgumentGradients` specifies `argnums`
# to get gradients for.
fn = gex.ArgumentGradients(tr, [1]).jit(f)(key, 0.3)
arg_grads = fn(key, 0.3)
print(arg_grads)

# Here's how you access choice gradients --
fn = gex.ChoiceGradients(tr).jit(f)(key, 0.3)
choices = {("m1",): 0.3}
choice_grads = fn(choices)
print(choice_grads)
```

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched) by the compositional tracing provided by `jax`.

This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - sans automatic insertion of staging annotations. The inference interfaces exposed handle lifting/splicing/jitting internally (manually).

## Tour

[Jump into the tour!](/tour.py)

# gex

> [**Ge**n](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [JA**X**](https://github.com/google/jax)

## Example

A straightforward encoding of Gen using zero-cost effect handling/tracing with `jax`.

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

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched). This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - without automatic insertion of staging annotations.

## Tour

[Jump into the tour!](/tour.py)

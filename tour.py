#####
##### Welcome to the `gex` tour!
#####

import jax
import gex

# A `gex` generative function is a pure Python function from
# `(PRNGKey, *args)` to `(PRNGKey, retval)`
#
# The programmer is free to use other JAX primitives, etc -- as desired.
#
# The models below are rather simplistic, but demonstrate
# proof of concept.
def g(key, x):
    key, m1 = gex.trace("m0", gex.Bernoulli)(key, x)
    return (key, m1)


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
fn = gex.Simulate().jit(f)(key, 0.3)
tr = fn(key, 0.3)
print(tr.get_choices())

# Here's how you access the `generate` GFI.
chm = {("m1",): 0.5}
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
print(tr.get_choices())
fn = gex.ChoiceGradients(tr).jit(f)(key, 0.3)
choices = {("m1",): 0.3}
choice_grads = fn(choices)
print(choice_grads)

# You can even re-use the jitted function to evaluate
# the gradient at different choice map points.
choices = {("m1",): 0.7, ("m2",): 0.3}
choice_grads = fn(choices)
print(choice_grads)

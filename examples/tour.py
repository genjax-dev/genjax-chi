#####
##### Welcome to the `genjax` tour!
#####

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


def f(key, x):
    key, m0 = gex.trace("m0", gex.Bernoulli)(key, x, shape=(3, 3))
    key, m1 = gex.trace("m1", gex.Normal)(key, shape=(5, 5))
    key, m2 = gex.trace("m2", gex.Laplace)(key, shape=(5, 5))
    key, m3 = gex.trace("m3", gex.Bernoulli)(key, m1)
    key, m4 = gex.trace("m4", g)(key, m1)
    return key, (2 * m1 * m2, m0, m3, m4)


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

# This just shows our raw (not yet desugared/codegen) syntax.
expr = gex.lift(f, key, 0.3)
print(expr)

# We can use our model function as a sampler.
key, v = gex.sample(f)(key, 0.3)
print((key, v))

# Here's how you access the `simulate` GFI.
tr = jax.jit(gex.simulate(f))(key, 0.3)
print(tr)

# Here's how you access the `generate` GFI.
chm = {("m1",): 0.3, ("m2",): 0.5}
w, tr = jax.jit(gex.generate(f))(chm, key, 0.3)
print((w, tr))

# Here's how you access the `arg_grad` interface.
arg_grad = jax.jit(gex.arg_grad(f, [1]))(tr, key, 0.3)
print(arg_grad)

# Here's how you access the `choice_grad` interface.
chm = {("m1",): 0.3, ("m2",): 0.5}
choice_grad, choices = jax.jit(gex.choice_grad(f))(tr, chm, key, 0.3)
print(choice_grad[("m1",)])

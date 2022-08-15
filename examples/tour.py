# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####
# Welcome to the `genjax` tour!
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
key, tr = jax.jit(gex.simulate(f))(key, 0.3)
print(tr.get_choices()[("m1",)])

# Here's how you access the `importance` GFI.
chm = gex.ChoiceMap({("m1",): 0.3, ("m2",): 0.5})
key, (w, tr) = jax.jit(gex.importance(f))(key, chm, 0.3)
print((w, tr))

# Here's how you access the `update` GFI.
chm = gex.ChoiceMap({("m1",): 0.2, ("m2",): 0.5})
key, (w, updated, discard) = jax.jit(gex.update(f))(key, tr, chm, 0.3)
print((w, updated, discard))

# Here's how you access the `arg_grad` interface.
key, arg_grad = jax.jit(gex.arg_grad(f, [1]))(key, tr, 0.3)
print(arg_grad)

# Here's how you access the `choice_grad` interface.
chm = gex.ChoiceMap({("m1",): 0.2, ("m2",): 0.5})
key, choice_grad = jax.jit(gex.choice_grad(f))(key, tr, chm, 0.3)
print(choice_grad)
print(choice_grad[("m1",)])

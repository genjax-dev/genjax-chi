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

import jax
import genjax

console = genjax.go_pretty()

# A `genjax` generative function is a pure Python function from
# `(PRNGKey, *args)` to `(PRNGKey, retval)`
#
# The programmer is free to use other JAX primitives, etc -- as desired.
#
# The models below are rather simplistic, but demonstrate
# proof of concept.


@genjax.gen
def h(key, x):
    key, m1 = genjax.trace("m0", genjax.Bernoulli)(key, (x,))
    return (key, m1)


@genjax.gen
def g(key, x):
    key, m1 = genjax.trace("m0", h)(key, (x,))
    return (key, m1)


@genjax.gen
def f(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli, shape=(3, 3))(key, (x,))
    key, m4 = genjax.trace("m4", g)(key, (x,))
    key, m5 = genjax.trace("m5", genjax.Normal)(key, (0.0, 1.0))
    return key, (2 * m0 * m4, m5)


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

# This just shows our raw (not yet desugared/codegen) syntax.
expr = jax.make_jaxpr(f)(key, 0.3)
console.graph(expr)

# Here's how you access the `simulate` GFI.
key, tr = jax.jit(genjax.simulate(f))(key, (0.3,))

# Here's how you access the `importance` GFI.
chm = genjax.ChoiceMap.new({("m0",): True, ("m4", "m0", "m0"): False})
key, (w, tr) = jax.jit(genjax.importance(f))(key, chm, (0.3,))
assert tr[("m4", "m0", "m0")] == False

# Here's how you access the `update` GFI.
jitted = jax.jit(genjax.update(f))

chm = genjax.ChoiceMap.new({("m0",): False})
key, (w, updated, discard) = jitted(key, tr, chm, (0.3,))

chm = genjax.ChoiceMap.new({("m4", "m0", "m0"): True})
print(chm)
key, (w, updated, discard) = jitted(key, tr, chm, (0.3,))
assert updated[("m4", "m0", "m0")] == True

# Here's how you access the `arg_grad` GFI.
jitted = jax.jit(genjax.arg_grad(f, argnums=1))
key, grad = jitted(key, tr, (0.3,))
print(grad)

# Here's how you access the `choice_grad` GFI.
jitted = jax.jit(genjax.choice_grad(f))
selected = genjax.Selection(["m5"])
key, choice_grad = jitted(key, tr, selected)
print(choice_grad["m5"])

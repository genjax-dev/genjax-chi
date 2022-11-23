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


# `go_pretty` provides a console object which supports
# pretty printing of `Pytree` objects.
console = genjax.go_pretty()


@genjax.gen
def f(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli, shape=(3, 3))(key, (x,))
    key, m1 = genjax.trace("m1", genjax.Normal, shape=(3, 3))(key, (0.0, 1.0))
    return key, m1 * m0


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)
jaxpr = jax.make_jaxpr(f)(key, 0.3)

key, tr = jax.jit(genjax.simulate(f))(key, (0.3,))
console.print(tr.get_retval())

maker = jax.make_jaxpr(genjax.update(f))
jaxpr = maker(
    key,
    tr,
    genjax.EmptyChoiceMap(),
    (genjax.Diff.new(0.3, genjax.NoChange),),
)
print(jaxpr)

jaxpr = maker(
    key,
    tr,
    genjax.ChoiceMap.new({("m0",): True}),
    (genjax.Diff.new(0.3, genjax.NoChange),),
)
print(jaxpr)

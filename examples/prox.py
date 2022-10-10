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
import genjax.experimental.prox as prox


console = genjax.go_pretty()

# This is showcasing auxiliary-variable inference strategies from
# an experimental implementation of `GenProx`.
#
# (GenProx, Alex Lew) https://github.com/probcomp/GenProx.jl


@genjax.gen(prox.ChoiceMapDistribution, selection=genjax.Selection(["y"]))
def proposal(key, target: prox.Target):
    key, v = genjax.trace("v", genjax.Normal)(key, (0.0, 1.0))
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, y = genjax.trace("y", genjax.Normal)(key, (0.0, 1.0))
    return key, y


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.Selection(["x"]),
    custom_q=proposal,
)
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, y = genjax.trace("y", genjax.Normal)(key, (0.0, 1.0))
    return (key, x**2 + y**2)


key = jax.random.PRNGKey(314159)
trace_type = genjax.get_trace_type(model)(key, ())
console.print(trace_type)
key, tr = genjax.simulate(model)(key, ())
console.print(tr)

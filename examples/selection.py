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


@genjax.gen
def submodel(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    key, y = genjax.trace("y", genjax.Bernoulli)(key, (0.3,))
    return key, x


@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    key, y = genjax.trace("y", genjax.Normal)(key, ())
    key, q = genjax.trace("q", submodel)(key, ())
    return key, x + y + q


def fn():
    hierarchical = genjax.Selection([("y",)])
    complement = hierarchical.complement()
    key = jax.random.PRNGKey(314159)
    key, tr = genjax.simulate(model)(key, ())
    chm = tr.get_choices()
    return chm, hierarchical.filter(chm), complement.filter(chm)


chm, hierarchical, complement = jax.jit(fn)()

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
import jax.numpy as jnp

import genjax
from genjax import typing


class TestDropArguments:
    def test_drop_arguments_as_kernel_in_map(self):
        @genjax.map_combinator(in_axes=(0,))
        @genjax.drop_arguments
        @genjax.gen
        @typing.typecheck
        def model(x: typing.FloatArray):
            y = genjax.normal(x, 1.0) @ "y"
            return y

        key = jax.random.PRNGKey(314159)
        chm = genjax.index_choice_map([0], {"y": jnp.array([5.0])})
        tr = model.simulate(key, (jnp.ones(5),))
        _, _, tr, _ = model.update(
            key, tr, chm, genjax.tree_diff_no_change((jnp.ones(5),))
        )
        v = tr.strip()["y"]
        assert v[0] == 5.0
        sel = genjax.index_select([0], genjax.select("y"))
        assert tr.project(sel) == genjax.normal.logpdf(5.0, 1.0, 1.0)

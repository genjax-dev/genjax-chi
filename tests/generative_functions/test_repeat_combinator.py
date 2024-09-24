# Copyright 2024 MIT Probabilistic Computing Project
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

import jax.numpy as jnp
from jax.random import PRNGKey

from genjax import ChoiceMapBuilder as C
from genjax import gen, normal


class TestRepeatCombinator:
    def test_repeat_combinator_importance(self):
        @gen
        def model():
            return normal(0.0, 1.0) @ "x"

        key = PRNGKey(314)
        tr, w = model.repeat(n=10).importance(key, C[1, "x"].set(3.0), ())
        assert normal.assess(C.v(tr.get_choices()[1, "x"]), (0.0, 1.0))[0] == w

    def test_repeat_matches_vmap(self):
        @gen
        def square(x):
            return x * x

        key = PRNGKey(314)
        repeat_retval = square.repeat(n=10)(2)(key)

        assert repeat_retval.shape == (10,), "We asked for and received 10 squares"

        assert jnp.array_equal(
            square.vmap()(jnp.repeat(2, 10))(key), repeat_retval
        ), "Repeat 10 times matches vmap with 10 equal inputs"

# Copyright 2023 MIT Probabilistic Computing Project
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
from genjax import ChoiceMap, Diff, IndexChange


class TestIncremental:
    def test_custom_incremental_rules(self):
        def default_rule(original_function, old_val, diff):
            (arr,) = diff
            return Diff.unknown_change(jnp.sum(arr.primal))

        @genjax.gen
        def no_choices_default_rule(v):
            return genjax.cache(
                "r",
                jnp.sum,
                custom_rule=default_rule,
            )(v)

        key = jax.random.PRNGKey(314159)
        original_v = jnp.ones(10)
        old_tr = no_choices_default_rule.simulate(key, (jnp.ones(10),))
        assert old_tr["r"] == jnp.sum(jnp.ones(10))
        new_v = original_v.at[3].set(3.0)

        new_tr, _, _, _ = old_tr.update(
            key,
            ChoiceMap.empty(),
            (Diff(new_v, IndexChange(jnp.array(3), 1.0)),),
        )
        assert new_tr["r"] == old_tr["r"] + 2.0

        def custom_rule(original_function, old_val, diff):
            (arr,) = diff
            idx_c: IndexChange = Diff.tree_tangent(arr)
            primal = Diff.tree_primal(arr)
            change = primal[idx_c.idx] - idx_c.old_val
            return Diff.unknown_change(old_val + change)

        @genjax.gen
        def no_choices_custom_rule(v):
            return genjax.cache(
                "r",
                jnp.sum,
                custom_rule=custom_rule,
            )(v)

        key = jax.random.PRNGKey(314159)
        original_v = jnp.ones(10)
        old_tr = no_choices_custom_rule.simulate(key, (jnp.ones(10),))
        assert old_tr["r"] == jnp.sum(jnp.ones(10))
        new_v = original_v.at[3].set(3.0)

        new_tr, _, _, _ = old_tr.update(
            key,
            ChoiceMap.empty(),
            (Diff(new_v, IndexChange(jnp.array(3), 1.0)),),
        )
        assert new_tr["r"] == old_tr["r"] + 2.0

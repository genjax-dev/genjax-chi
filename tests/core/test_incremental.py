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

import genjax
import jax
import jax.numpy as jnp
from genjax import ChoiceMap, Diff, IndexChange


class TestIncremental:
    def custom_rule(self, original_function, old_val, diff):
        (arr,) = diff
        return Diff.unknown_change(jnp.sum(arr.primal))

    def test_custom_incremental_rules(self):
        @genjax.gen
        def no_choices(v):
            return genjax.cache(
                "r",
                jnp.sum,
                custom_rule=self.custom_rule,
            )(v)

        key = jax.random.PRNGKey(314159)
        original_v = jnp.ones(10)
        old_tr = no_choices.simulate(key, (jnp.ones(10),))
        assert old_tr["r"] == jnp.sum(jnp.ones(10))
        new_v = original_v.at[3].set(3.0)

        new_tr, _, _, _ = old_tr.update(
            key,
            ChoiceMap.empty(),
            (Diff(new_v, IndexChange(jnp.array(3), 1.0)),),
        )
        assert new_tr["r"] == old_tr["r"] + 2.0

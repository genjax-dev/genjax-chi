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
import numpy as np
import genjax

console = genjax.go_pretty()


@genjax.gen(genjax.Unfold, max_length=10)
def fn(key, prev_state):
    key, new = genjax.trace("z", genjax.Normal)(key, (prev_state, 1.0))
    return key, new


obs = genjax.ChoiceMap.new({("z",): np.ones(5)})

key = jax.random.PRNGKey(314159)
key, (w, tr) = genjax.importance(fn)(key, obs, (10, 0.1))
console.print(tr.get_choices())

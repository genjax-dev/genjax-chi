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
def random_walk(key, prev):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    return (key, x + prev)


unfold = genjax.UnfoldCombinator(random_walk, 10)

key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(unfold))(key, (0.3,))
print(tr.get_retval())

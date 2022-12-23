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

import jax.tree_util as jtu


def map_update(selection, tau):
    def _inner(key, trace):
        args = trace.get_args()
        gen_fn = trace.get_gen_fn()
        key, forward_gradient_trie = gen_fn.choice_grad(key, trace, selection)
        forward_values, _ = selection.filter(trace)
        forward_values = forward_values.strip()
        forward_values = jtu.tree_map(
            lambda v1, v2: v1 + tau * v2,
            forward_values,
            forward_gradient_trie,
        )
        key, (w, new_trace, _) = gen_fn.update(
            key, trace, forward_values, args
        )
        return key, (new_trace, True)

    return _inner

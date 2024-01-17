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

from dataclasses import dataclass

import jax
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import FloatArray


@dataclass
class MapUpdate(Pytree):
    selection: Selection
    tau: FloatArray

    def flatten(self):
        return (self.tau,), (self.selection,)

    def _grad_step_no_none(self, v1, v2):
        if v2 is None:
            return v1
        else:
            return v1 + self.tau * v2

    def apply(self, key, trace):
        args = trace.get_args()
        gen_fn = trace.get_gen_fn()
        argdiffs = jtu.tree_map(Diff.no_change, args)
        key, sub_key = jax.random.split(key)
        forward_gradient_trie = gen_fn.choice_grad(sub_key, trace, self.selection)
        forward_values = self.selection.filter(trace.strip())
        forward_values = forward_values.strip()
        forward_values = jtu.tree_map(
            self._grad_step_no_none,
            forward_values,
            forward_gradient_trie,
        )
        (_, _, new_trace, _) = gen_fn.update(key, trace, forward_values, argdiffs)
        return (new_trace, True)

    def __call__(self, key, trace):
        return self.apply(key, trace)

# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import jax.tree_util as jtu

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import static_check_is_concrete


@dataclass
class PytreeConst(Pytree):
    const: Any

    def flatten(self):
        return (), (self.const)


def const(v):
    return PytreeConst(v)


def tree_map_static_dynamic(v):
    def _inner(v):
        if static_check_is_concrete(v):
            return v
        else:
            return PytreeConst(v)

    return jtu.tree_map(_inner, v)

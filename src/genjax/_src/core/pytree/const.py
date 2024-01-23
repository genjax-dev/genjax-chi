# Copyright 2023 MIT Probabilistic Computing Project
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


import jax.tree_util as jtu
import rich.tree as rich_tree

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any, static_check_is_concrete


class PytreeConst(Pytree):
    const: Any = Pytree.static()

    def __rich_tree__(self):
        return rich_tree.Tree(f"[bold](PytreeConst) {self.const}")


def const(v):
    # The value must be concrete!
    # It cannot be a JAX traced value.
    assert static_check_is_concrete(v)
    if isinstance(v, PytreeConst):
        return v
    else:
        return PytreeConst(v)


def tree_map_const(v):
    def _inner(v):
        if isinstance(v, PytreeConst):
            return v
        elif static_check_is_concrete(v):
            return PytreeConst(v)
        else:
            return v

    return jtu.tree_map(
        _inner,
        v,
        is_leaf=lambda v: isinstance(v, PytreeConst),
    )


def tree_map_collapse_const(v):
    def _inner(v):
        if isinstance(v, PytreeConst):
            return v.const
        else:
            return v

    return jtu.tree_map(
        _inner,
        v,
        is_leaf=lambda v: isinstance(v, PytreeConst),
    )

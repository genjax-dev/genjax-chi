# Copyright 2022 The MIT Probabilistic Computing Project
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

import dataclasses
from typing import Any
from typing import List

import jax.tree_util as jtu

from genjax._src.core.interpreters.cps import Cell
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import IntArray


#######################################
# Change type lattice and propagation #
#######################################

#####
# Changes
#####


class Change(Pytree):
    def coerce_to_coarse(self):
        return UnknownChange


# These two classes are the bottom and top of the change lattice.
# Unknown change represents complete lack of information about
# the change to a value.
#
# No change represents complete information about the change to a value
# (namely, that it is has not changed).


class _UnknownChange(Change):
    def flatten(self):
        return (), ()


UnknownChange = _UnknownChange()


class _NoChange(Change):
    def flatten(self):
        return (), ()

    def coerce_to_coarse(self):
        return self


NoChange = _NoChange()


class IntChange(Change):
    dv: IntArray

    def flatten(self):
        return (self.dv,), ()


#####
# Diffs
#####

# Diffs carry values (often tracers, which are already abstract) and
# changes - which are also often tracers - but with extra metadata which
# can furnish additional optimizations in generative function languages.


@dataclasses.dataclass
class Diff(Cell):
    val: Any
    change: Change

    def flatten(self):
        return (self.val, self.change), ()

    @classmethod
    def new(cls, val, change: Change = UnknownChange):
        if isinstance(val, Diff):
            return Diff.new(val.get_val(), change=val.get_change())
        return Diff(val, change)

    @classmethod
    def no_change(cls, v):
        return Diff.new(v, change=NoChange)

    def get_change(self):
        return self.change

    def get_val(self):
        return self.val

    @classmethod
    def tree_map_diff(cls, tree, change_value):
        return jtu.tree_map(lambda v: Diff.new(v, change=change_value), tree)


def check_is_diff(v):
    return isinstance(v, Diff) or isinstance(v, Cell)


def tree_strip_diff(tree):
    def _check(v):
        return isinstance(v, Diff)

    def _inner(v):
        if isinstance(v, Diff):
            return v.get_val()
        else:
            return v

    return jtu.tree_map(_inner, tree, is_leaf=_check)


def fallback_diff_rule(prim: Any, incells: List[Diff], **params):
    in_vals = list(map(lambda v: v.get_val(), incells))
    out = prim.bind(*in_vals, **params)
    if all(map(lambda v: v.get_change() == NoChange, incells)):
        new_out = Diff.tree_map_diff(out, NoChange)
    else:
        new_out = Diff.tree_map_diff(out, UnknownChange)
    if not prim.multiple_results:
        new_out = [new_out]
    return new_out


def check_no_change(diff):
    return isinstance(diff.get_change(), _NoChange)

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

from genjax.core.propagate import Cell
from genjax.core.propagate import PropagationRules
from genjax.core.propagate import default_propagation_rules
from genjax.core.pytree import Pytree
from genjax.core.staging import get_shaped_aval
from genjax.core.typing import IntegerTensor


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
    dv: IntegerTensor

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

    def __init__(self, aval, val, change):
        super().__init__(aval)
        self.val = val
        self.change = change

    def flatten(self):
        return (self.val, self.change), (self.aval,)

    def __lt__(self, other):
        return self.bottom() and other.top()

    def top(self):
        return self.val is not None

    def bottom(self):
        return self.val is None

    def join(self, other):
        if other.bottom():
            return self
        else:
            return other

    @classmethod
    def new(cls, val, change: Change = UnknownChange):
        if isinstance(val, Diff):
            return Diff.new(val.get_val(), change=val.get_change())
        aval = get_shaped_aval(val)
        return Diff(aval, val, change)

    @classmethod
    def unknown(cls, aval):
        return Diff(aval, None, UnknownChange)

    @classmethod
    def no_change(cls, v):
        return Diff.new(v, change=NoChange)

    def get_change(self):
        return self.change

    def get_val(self):
        return self.val


def check_is_diff(v):
    return isinstance(v, Diff) or isinstance(v, Cell)


def strip_diff(diff):
    return diff.get_val()


def fallback_diff_rule(
    prim: Any, incells: List[Diff], outcells: Any, **params
):
    if all(map(lambda v: v.top(), incells)):
        in_vals = list(map(lambda v: v.get_val(), incells))
        out = prim.bind(*in_vals, **params)
        if all(map(lambda v: v.get_change() == NoChange, incells)):
            new_out = jtu.tree_map(lambda v: Diff.new(v, change=NoChange), out)
        else:
            new_out = jtu.tree_map(lambda v: Diff.new(v), out)
        if not prim.multiple_results:
            new_out = [new_out]
    else:
        new_out = outcells
    return incells, new_out, None


def check_no_change(diff):
    return diff.get_change() == NoChange


diff_propagation_rules = PropagationRules(
    fallback_diff_rule, default_propagation_rules.get_rule_set()
)

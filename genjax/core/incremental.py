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

"""
An incremental computation interpreter which supports user-defined
incremental plugins (primitives and incremental derivatives).
"""

import functools
from genjax.core.propagating import Cell, get_shaped_aval

#####
# Coarse incremental propagation
#####


class CoarseIncremental(Cell):
    def __init__(self, aval, val):
        super().__init__(aval)
        self.val = val

    def __lt__(self, other):
        return self.bottom() and other.top()

    def top(self):
        return self.val is False

    def bottom(self):
        return self.val is True

    def join(self, other):
        if other.bottom():
            return other
        else:
            return self

    @classmethod
    def new(cls, val):
        aval = get_shaped_aval(val)
        return CoarseIncremental(aval, val)

    @classmethod
    def unknown(cls, aval):
        return CoarseIncremental(aval, True)

    def flatten(self):
        return (self.val,), (self.aval,)

    @classmethod
    def unflatten(cls, data, xs):
        return CoarseIncremental(data[0], xs[0])


def default_coarse_incremental_rule(prim, invals, outvals, **params):
    """
    Default coarse_incremental rule that falls back to re-evaluation with
    changed arguments.
    """
    if all(outval.bottom() for outval in outvals):
        if all(inval.top() for inval in invals):
            vals = [inval.val for inval in invals]
            ans = prim.bind(*vals, **params)
            if not prim.multiple_results:
                ans = [ans]
            outvals = safe_map(CoarseIncremental.new, ans)
        return invals, outvals, None
    if any(outval.bottom() for outval in outvals):
        return invals, outvals, None
    raise NotImplementedError(f"No registered inverse for `{prim}`.")


class CoarseIncrementalDict(object):
    """
    Default rules dictionary that uses a default rule for
    incremental computation.
    """

    def __init__(self):
        self.rules = {}

    def __getitem__(self, prim):
        if prim not in self.rules:
            self[prim] = functools.partial(
                default_coarse_incremental_rule, prim
            )
        return self.rules[prim]

    def __setitem__(self, prim, val):
        self.rules[prim] = val

    def __contains__(self, prim):
        return prim in self.rules


coarse_incremental_rules = CoarseIncrementalDict()

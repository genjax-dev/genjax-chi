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

import jax.core as jc
from jax.util import safe_map
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.builtin.intrinsics import gen_fn_p
from genjax.builtin.tree import Tree
from genjax.core.tracetypes import TraceType, Reals, Integers, Finite
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

#####
# BuiltinTraceType
#####


@dataclass
class BuiltinTraceType(TraceType):
    tree: Tree
    return_type: TraceType

    def flatten(self):
        return (), (self.tree, self.return_type)

    @classmethod
    def unflatten(cls, xs, data):
        return BuiltinTraceType(*xs, *data)

    def get_choices(self):
        return self.tree

    def get_choices_shallow(self):
        return self.tree.get_choices_shallow()

    def get_rettype(self):
        return self.return_type

    def overload_pprint(self, **kwargs):
        return pp.concat(
            [
                gpp._pformat(self.tree, **kwargs),
                pp.brk(""),
                pp.text("return_type -> "),
                gpp._pformat(self.return_type, **kwargs),
            ]
        )

    def subseteq(self, other):
        if not isinstance(other, BuiltinTraceType):
            return False, self
        else:
            check = True
            tree = Tree({})
            for (k, v) in self.get_choices_shallow():
                if other.tree.has_choice(k):
                    sub = other.tree[k]
                    subcheck, mismatch = v.subseteq(sub)
                    if not subcheck:
                        tree[k] = mismatch
                else:
                    check = False
                    tree[k] = (v, None)

            for (k, v) in other.get_choices_shallow():
                if not self.tree.has_choice(k):
                    check = False
                    tree[k] = (None, v)
            return check, tree

    def __subseteq__(self, other):
        check, _ = self.subseteq(other)
        return check

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)


def get_trace_type(jaxpr: jc.ClosedJaxpr):
    env = {}
    trace_type = Tree({})

    def read(var):
        if type(var) is jc.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.jaxpr.invars, jaxpr.in_avals)
    safe_map(write, jaxpr.jaxpr.constvars, jaxpr.literals)

    for eqn in jaxpr.eqns:
        if eqn.primitive == gen_fn_p:
            gen_fn = eqn.params["gen_fn"]
            addr = eqn.params["addr"]
            invals = safe_map(read, eqn.invars)
            key = invals[0]
            args = tuple(invals[1:])
            ty = gen_fn.get_trace_type(key, args, **eqn.params)
            trace_type[addr] = ty
        outvals = safe_map(lambda v: v.aval, eqn.outvars)
        safe_map(write, eqn.outvars, outvals)

    key = jaxpr.out_avals[0]
    return_type = tuple(map(lift, jaxpr.out_avals[1:]))
    return BuiltinTraceType(trace_type, return_type)


# Lift Python values to the Trace Types lattice.
def lift(v, shape=()):
    if v == jnp.int32:
        return Integers(shape)
    if v == jnp.float32:
        return Reals(shape)
    if v == bool:
        return Finite(shape, 2)
    if isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
        ty = lift(v.dtype, shape=v.shape)
        return ty
    elif isinstance(v, jc.ShapedArray):
        ty = lift(v.dtype, shape=v.shape)
        return ty

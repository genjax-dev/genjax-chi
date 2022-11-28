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

import jax
import jax.core as jc
import jax.numpy as jnp
import numpy as np
from jax.util import safe_map

from genjax.core.hashabledict import HashableDict
from genjax.core.hashabledict import hashabledict
from genjax.core.tracetypes import Finite
from genjax.core.tracetypes import Integers
from genjax.core.tracetypes import Reals
from genjax.core.tracetypes import TraceType
from genjax.generative_functions.builtin.intrinsics import gen_fn_p


@dataclass
class BuiltinTraceType(TraceType):
    inner: HashableDict
    return_type: TraceType

    def flatten(self):
        return (), (self.inner, self.return_type)

    def get_leaf_value(self):
        raise Exception("BuiltinTraceType is not a leaf choice tree.")

    def has_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.has_subtree(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.get_subtree(rest)
            else:
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other[k]
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        if isinstance(other, BuiltinTraceType):
            return BuiltinTraceType(new, other.get_rettype())
        else:
            return BuiltinTraceType(new, self.get_rettype())

    def get_rettype(self):
        return self.return_type

    def subseteq(self, other):
        if not isinstance(other, BuiltinTraceType):
            return False, self
        else:
            check = True
            tree = dict()
            for (k, v) in self.get_subtrees_shallow():
                if k in other.inner:
                    sub = other.inner[k]
                    subcheck, mismatch = v.subseteq(sub)
                    if not subcheck:
                        tree[k] = mismatch
                else:
                    check = False
                    tree[k] = (v, None)

            for (k, v) in other.get_subtrees_shallow():
                if k not in self.inner:
                    check = False
                    tree[k] = (None, v)
            return check, tree

    def __subseteq__(self, other):
        check, _ = self.subseteq(other)
        return check


def get_trace_type(jaxpr: jc.ClosedJaxpr):
    env = {}
    trace_type = dict()

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
        return lift(v.dtype, shape=v.shape)
    if isinstance(v, jax.ShapeDtypeStruct):
        return lift(v.dtype, shape=v.shape)
    elif isinstance(v, jc.ShapedArray):
        return lift(v.dtype, shape=v.shape)

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
import jax.tree_util as jtu
from jax.util import safe_map

from genjax._src.core.tracetypes import Bottom
from genjax._src.core.tracetypes import Finite
from genjax._src.core.tracetypes import Integers
from genjax._src.core.tracetypes import Reals
from genjax._src.core.tracetypes import TraceType
from genjax._src.core.typing import static_check_is_array
from genjax._src.generative_functions.builtin.intrinsics import gen_fn_p
from genjax._src.generative_functions.builtin.trie import Trie


@dataclass
class BuiltinTraceType(TraceType):
    trie: Trie
    return_type: TraceType

    def flatten(self):
        return (), (self.trie, self.return_type)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return Bottom()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()

    def merge(self, other):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other.get_subtree(k)
                trie[k] = v.merge(sub)
            else:
                trie[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not trie.has_subtree(k):
                trie[k] = v
        if isinstance(other, BuiltinTraceType):
            return BuiltinTraceType(trie, other.get_rettype())
        else:
            return BuiltinTraceType(trie, self.get_rettype())

    def get_rettype(self):
        return self.return_type

    def on_support(self, other):
        if not isinstance(other, BuiltinTraceType):
            return False, self
        else:
            check = True
            trie = Trie.new()
            for (k, v) in self.get_subtrees_shallow():
                if k in other.trie:
                    sub = other.trie[k]
                    subcheck, mismatch = v.on_support(sub)
                    if not subcheck:
                        trie[k] = mismatch
                else:
                    check = False
                    trie[k] = (v, None)

            for (k, v) in other.get_subtrees_shallow():
                if k not in self.trie:
                    check = False
                    trie[k] = (None, v)
            return check, trie

    def __check__(self, other):
        check, _ = self.on_support(other)
        return check


######
# Typing interpreter
######


def trace_typer(jaxpr: jc.ClosedJaxpr):
    env = {}
    trace_type = Trie.new()

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
            tree_in = eqn.params["tree_in"]
            addr = eqn.params["addr"]
            invals = safe_map(read, eqn.invars)
            gen_fn, args = jtu.tree_unflatten(tree_in, invals)
            ty = gen_fn.get_trace_type(*args, **eqn.params)
            trace_type[addr] = ty
        outvals = safe_map(lambda v: v.aval, eqn.outvars)
        safe_map(write, eqn.outvars, outvals)

    return_type = tuple(map(lift, jaxpr.out_avals))

    # Collapse single arity returns.
    if len(return_type) == 1:
        return_type = return_type[0]

    return BuiltinTraceType(trace_type, return_type)


# Lift Python values to the trace type lattice.
def lift(v, shape=()):
    if v == jnp.int32:
        return Integers(shape)
    if v == jnp.float32:
        return Reals(shape)
    if v == bool:
        return Finite(shape, 2)
    if static_check_is_array(v):
        return lift(v.dtype, shape=v.shape)
    if isinstance(v, jax.ShapeDtypeStruct):
        return lift(v.dtype, shape=v.shape)
    elif isinstance(v, jc.ShapedArray):
        return lift(v.dtype, shape=v.shape)

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

import functools
from dataclasses import dataclass

import jax.core as jc
import jax.tree_util as jtu
from jax.util import safe_map

from genjax._src.core.datatypes.tracetypes import Bottom
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.tracetypes import tt_lift
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import stage
from genjax._src.generative_functions.builtin.builtin_primitives import gen_fn_p


@dataclass
class BuiltinTraceType(TraceType):
    trie: Trie
    retval_type: TraceType

    def flatten(self):
        return (), (self.trie, self.retval_type)

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

    def get_rettype(self):
        return self.retval_type

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


def trace_typing(jaxpr: jc.ClosedJaxpr, flat_in, consts):
    # Simple environment, nothing fancy required.
    env = {}
    inner_trace_type = Trie.new()

    def read(var):
        if type(var) is jc.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, flat_in)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        if eqn.primitive == gen_fn_p:
            in_tree = eqn.params["in_tree"]
            addr = eqn.params["addr"]
            invals = safe_map(read, eqn.invars)
            gen_fn, *args = jtu.tree_unflatten(in_tree, invals)
            ty = gen_fn.get_trace_type(*args, **eqn.params)
            inner_trace_type[addr] = ty
        outvals = safe_map(lambda v: v.aval, eqn.outvars)
        safe_map(write, eqn.outvars, outvals)

    return safe_map(read, jaxpr.outvars), inner_trace_type


def trace_type_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(*args):
        closed_jaxpr, (flat_in, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out, inner_tt = trace_typing(jaxpr, flat_in, consts)
        flat_out = list(map(lambda v: tt_lift(v), flat_out))
        if flat_out:
            rettypes = jtu.tree_unflatten(out_tree, flat_out)
        else:
            rettypes = tt_lift(None)
        return BuiltinTraceType(inner_tt, rettypes)

    return _inner

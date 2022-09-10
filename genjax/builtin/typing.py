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
from genjax.builtin.intrinsics import gen_fn_p
from genjax.builtin.tree import Tree


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
            ty = gen_fn.get_trace_type(*invals, **eqn.params)
            trace_type[addr] = ty
        outvals = safe_map(lambda v: v.aval, eqn.outvars)
        safe_map(write, eqn.outvars, outvals)

    return trace_type

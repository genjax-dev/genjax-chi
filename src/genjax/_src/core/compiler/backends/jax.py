# Copyright 2024 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend.core import Jaxpr
from jax.lax import cond_p, scan, scan_p

from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.pjax import sample_p
from genjax._src.core.compiler.staging import stage
from genjax._src.core.typing import Any, Callable, PRNGKey

####################
# PJAX Interpreter #
####################


@dataclass
class JAXCInterpreter:
    key: PRNGKey

    def eval_jaxpr_jaxc(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            if eqn.primitive == sample_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + invals
                jax_impl = params["jax_impl"]
                self.key, sub_key = jrand.split(self.key)
                outvals = jtu.tree_leaves(
                    jax_impl(sub_key, *args),
                )
            elif eqn.primitive == cond_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                index_val, ops_vals = invals[0], invals[1:]

                def new_branch(key, closed_jaxpr, *args):
                    interpreter = JAXCInterpreter(key)
                    jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
                    return interpreter.eval_jaxpr_jaxc(
                        jaxpr,
                        consts,
                        list(args),
                    )

                self.key, sub_key = jrand.split(self.key)
                new_branches = [
                    lambda *args: new_branch(sub_key, branch_closed_jaxpr, *args)
                    for branch_closed_jaxpr in branch_closed_jaxprs
                ]
                outvals = new_branches[0](*ops_vals)

            # We replace the original scan with a new scan
            # that calls the interpreter on the scan body,
            # carries the key through and evolves it.
            elif eqn.primitive == scan_p:
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = jax_util.split_list(
                    invals, [num_consts, num_carry]
                )

                def new_flat_scan(carry, scanned_in):
                    (key, original_carries) = carry
                    (idx, original_scanned_in) = scanned_in
                    sub_key = jrand.fold_in(key, idx)
                    interpreter = JAXCInterpreter(sub_key)
                    outvals = interpreter.eval_jaxpr_jaxc(
                        body_jaxpr.jaxpr,
                        const_vals,
                        jtu.tree_leaves(
                            (original_carries, original_scanned_in),
                        ),
                    )
                    _, carry_out, scanned_out = jax_util.split_list(
                        outvals, [num_consts, num_carry]
                    )
                    return (key, carry_out), scanned_out

                self.key, sub_key = jrand.split(self.key)
                fold_idxs = jnp.arange(length)
                (_, flat_carry_out), scanned_out = scan(
                    new_flat_scan,
                    (sub_key, carry_vals),
                    (fold_idxs, xs_vals),
                    length=length,
                    reverse=reverse,
                )
                outvals = jtu.tree_leaves(
                    (flat_carry_out, scanned_out),
                )

            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_jaxc(
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def pjax_to_jax(
    key: PRNGKey,
    f: Callable[..., Any],
):
    @functools.wraps(f)
    def wrapped(*args):
        interpreter = JAXCInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped

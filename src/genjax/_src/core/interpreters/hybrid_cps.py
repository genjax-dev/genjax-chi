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

from abc import abstractmethod

import jax
import jax.core as jc
import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend import source_info_util as src_util

from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import ArrayLike, Callable, List, typecheck

register_exclusion(__file__)


class CPSPrimitive(Pytree):
    @abstractmethod
    def default_call(self, *args):
        raise NotImplementedError

    @abstractmethod
    def handle(self, kont: Callable, *args):
        raise NotImplementedError

    @typecheck
    def __call__(self, *args):
        return cont_primitive(self, *args)


break_p = InitialStylePrimitive("cps_breakpoint")


@typecheck
def cont_primitive(cont_prim: CPSPrimitive, *args):
    def _cont_prim_call(cont_prim: CPSPrimitive, *args):
        return cont_prim.default_call(*args)

    return initial_style_bind(break_p)(_cont_prim_call)(cont_prim, *args)


##########################
# Hybrid CPS interpreter #
##########################


@Pytree.dataclass
class HybridCPSInterpreter(Pytree):
    @staticmethod
    def _eval_jaxpr_hybrid_cps(
        jaxpr: jc.Jaxpr,
        consts: List[ArrayLike],
        flat_args: List[ArrayLike],
        out_tree,
    ):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, flat_args)

        # Hybrid CPS evaluation.
        def eval_jaxpr_iterate_cps(
            eqns,
            env: Environment,
            invars,
            flat_args,
        ):
            jax_util.safe_map(env.write, invars, flat_args)

            for eqn_idx, eqn in list(enumerate(eqns)):
                with src_util.user_context(eqn.source_info.traceback):
                    invals = jax_util.safe_map(env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    args = subfuns + invals

                    if eqn.primitive == break_p:
                        env = env.copy()

                        # Create dual continuation.
                        def _kont(*args):
                            leaves = jtu.tree_leaves(args)
                            return eval_jaxpr_iterate_cps(
                                eqns[eqn_idx + 1 :],
                                env,
                                eqn.outvars,
                                leaves,
                            )

                        in_tree = params["in_tree"]
                        num_consts = params["num_consts"]
                        cps_prim, *args = jtu.tree_unflatten(in_tree, args[num_consts:])

                        return cps_prim.handle(
                            _kont,
                            *args,
                        )

                    else:
                        outs = eqn.primitive.bind(*args, **params)

                if not eqn.primitive.multiple_results:
                    outs = [outs]

                jax_util.safe_map(
                    env.write,
                    eqn.outvars,
                    outs,
                )

            out_values = jax.util.safe_map(
                env.read,
                jaxpr.outvars,
            )
            return jtu.tree_unflatten(out_tree(), out_values)

        return eval_jaxpr_iterate_cps(
            jaxpr.eqns,
            env,
            jaxpr.invars,
            flat_args,
        )

    @staticmethod
    def interactive(f):
        def _inner(*args):
            closed_jaxpr, (flat_args, _, out_tree) = stage(f)(*args)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            return HybridCPSInterpreter._eval_jaxpr_hybrid_cps(
                jaxpr,
                consts,
                flat_args,
                out_tree,
            )

        return _inner


def interactive(f):
    return HybridCPSInterpreter.interactive(f)

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

import abc
import functools

import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend.core import Jaxpr, Primitive

from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, Callable

########################
# Stateful interpreter #
########################


class StatefulHandler:
    @abc.abstractmethod
    def handles(self, primitive: Primitive) -> bool:
        pass

    @abc.abstractmethod
    def dispatch(
        self,
        primitive: Primitive,
        *args,
        **kwargs,
    ) -> list[Any]:
        pass


@Pytree.dataclass
class StatefulInterpreter(Pytree):
    def eval_jaxpr_stateful(
        self,
        stateful_handler,
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
            # Allow the stateful handler to handle the primitive.
            if stateful_handler.handles(eqn.primitive):
                outvals = stateful_handler.dispatch(eqn.primitive, *args, **params)
            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, stateful_handler, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_stateful(
            stateful_handler,
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def stateful(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(stateful_handler: StatefulHandler, *args):
        interpreter = StatefulInterpreter()
        return interpreter.run_interpreter(
            stateful_handler,
            f,
            *args,
        )

    return wrapped

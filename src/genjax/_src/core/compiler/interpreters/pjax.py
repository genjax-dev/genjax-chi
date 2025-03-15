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

import jax.random as jrand
import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend.core import Jaxpr

from genjax._src.core.compiler.initial_style_primitive import (
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.compiler.interpreters.common import Environment
from genjax._src.core.compiler.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, Callable, PRNGKey

######################
# Sampling primitive #
######################

sample_p = InitialStylePrimitive("sample")


def sample_binder(
    jax_impl: Callable[[PRNGKey, Any], Any],
    **kwargs,
):
    return initial_style_bind(sample_p, **kwargs)(jax_impl)


####################
# PJAX Interpreter #
####################


@dataclass
class PJAXInterpreter(Pytree):
    key: PRNGKey

    def _eval_jaxpr_pjax(
        self,
        _jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        jax_util.safe_map(env.write, _jaxpr.constvars, consts)
        jax_util.safe_map(env.write, _jaxpr.invars, args)
        for eqn in _jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            outvals = eqn.primitive.bind(*args, **params)
            if eqn.primitive == sample_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + invals
                _impl = params["_impl"]
                self.key, sub_key = jrand.split(self.key)
                outvals = _impl(sub_key, *args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, _jaxpr.outvars)

    def run_interpreter(self, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_pjax(
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def pjax(f: Callable[..., Any], key: PRNGKey):
    @functools.wraps(f)
    def wrapped(*args):
        interpreter = PJAXInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped

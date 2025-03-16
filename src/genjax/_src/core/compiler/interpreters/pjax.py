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
from jax import vmap
from jax.extend.core import Jaxpr

from genjax._src.core.compiler.initial_style_primitive import (
    InitialStylePrimitive,
    NotEliminatedException,
    initial_style_bind,
)
from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.staging import stage
from genjax._src.core.typing import Any, Callable, PRNGKey

######################
# Sampling primitive #
######################

sample_p = InitialStylePrimitive("sample")


def sample_binder(
    jax_impl: Callable[[PRNGKey, Any], Any],
    **kwargs,
):
    def sampler(*args):
        def keyless_jax_impl(*args):
            return jax_impl(jrand.PRNGKey(1), *args)

        def raise_exception():
            raise NotEliminatedException(
                "JAX is attempting to invoke the implementation of a sampler defined using the `sample_p` primitive in your function.\n\nEliminate `sample_p` in `your_fn` by using the `genjax.pjax(your_fn, key: PRNGKey)(*your_args)` transformation, which allows you to use the JAX implementation of the sampler."
            )

        def keyless_batch_impl(vector_args, batch_axes):
            v = initial_style_bind(
                sample_p,
                jax_impl=vmap(
                    jax_impl,
                    in_axes=(None, *batch_axes),
                ),
                raise_exception=raise_exception,
            )(vmap(keyless_jax_impl, in_axes=batch_axes))(*vector_args)
            return (v,), (0,)

        return initial_style_bind(
            sample_p,
            jax_impl=jax_impl,
            batch=keyless_batch_impl,
            raise_exception=raise_exception,
            **kwargs,
        )(keyless_jax_impl)(*args)

    return sampler


####################
# PJAX Interpreter #
####################


@dataclass
class PJAXInterpreter:
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
            if eqn.primitive == sample_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + invals
                jax_impl = params["jax_impl"]
                self.key, sub_key = jrand.split(self.key)
                outvals = jtu.tree_leaves(
                    jax_impl(sub_key, *args),
                )
            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, _jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
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

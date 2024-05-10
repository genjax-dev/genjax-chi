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
from functools import wraps

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import core as jc
from jax import util as jax_util
from jax.extend import source_info_util as src_util
from jax.interpreters import ad as jax_autodiff

from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    List,
    PRNGKey,
    Tuple,
    typecheck,
)

###################
# ADEV primitives #
###################


class ADEVPrimitive(Pytree):
    """
    An `ADEVPrimitive` is a primitive sampler equipped with a JVP gradient estimator strategy. These objects support forward sampling, but also come equipped with a strategy that interacts with ADEV's AD transformation to return a JVP estimate.
    """

    @abstractmethod
    def sample(self, key, *args):
        raise NotImplementedError

    @abstractmethod
    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,  # Pytree with Dual leaves.
        konts: Tuple[Callable, Callable],
    ) -> "Dual":
        pass

    @typecheck
    def __call__(self, *args):
        return sample_primitive(self, *args)


####################
# Sample intrinsic #
####################


sample_p = InitialStylePrimitive("sample")


@typecheck
def sample_primitive(adev_prim: ADEVPrimitive, *args, key=jax.random.PRNGKey(0)):
    def _adev_prim_call(adev_prim, *args):
        # When used for abstract tracing, value of the key doesn't matter.
        # However, we support overloading the key for other transformations,
        # which will rely on the default semantics of `initial_style_bind`,
        # which is to call this function -- then, the value of the key will matter.
        v = adev_prim.sample(key, *args)
        return v

    return initial_style_bind(sample_p)(_adev_prim_call)(
        adev_prim,
        *args,
    )


####################
# ADEV interpreter #
####################


@Pytree.dataclass
class Dual(Pytree):
    primal: Any
    tangent: Any

    @staticmethod
    def tree_pure(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v
            else:
                return Dual(v, jnp.zeros_like(v))

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_dual(primals, tangents):
        return jtu.tree_map(lambda v1, v2: Dual(v1, v2), primals, tangents)

    @staticmethod
    def tree_primal(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v.primal
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_tangent(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v.tangent
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_leaves(v):
        v = Dual.tree_pure(v)
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_unzip(v):
        primals = jtu.tree_leaves(Dual.tree_primal(v))
        tangents = jtu.tree_leaves(Dual.tree_tangent(v))
        return tuple(primals), tuple(tangents)


@Pytree.dataclass
class ADInterpreter(Pytree):
    """The `ADInterpreter` takes a `Jaxpr`,
    propagates dual numbers through it, while also performing a CPS transformation,
    to compute forward mode AD.

    When this interpreter hits
    the `sample_p` primitive, it creates a pair of continuation closures which is passed to the gradient strategy which the primitive is using.
    """

    @staticmethod
    def flat_unzip(duals: List):
        primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in duals)
        return list(primals), list(tangents)

    @staticmethod
    def _eval_jaxpr_adev_jvp(
        key: PRNGKey,
        jaxpr: jc.Jaxpr,
        consts: List[ArrayLike],
        flat_duals: List[Dual],
    ):
        dual_env = Environment()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, Dual.tree_pure(consts))
        jax_util.safe_map(dual_env.write, jaxpr.invars, flat_duals)

        # TODO: Pure evaluation.
        def eval_jaxpr_iterate_pure(key, eqns, pure_env, invars, flat_args):
            jax_util.safe_map(pure_env.write, invars, flat_args)
            for eqn in eqns:
                in_vals = jax_util.safe_map(pure_env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals
                if eqn.primitive is sample_p:
                    pass
                else:
                    outs = eqn.primitive.bind(*args, **params)
                if not eqn.primitive.multiple_results:
                    outs = [outs]
                jax_util.safe_map(pure_env.write, eqn.outvars, outs)

            return jax_util.safe_map(pure_env.read, jaxpr.outvars)

        # Dual evaluation.
        def eval_jaxpr_iterate_dual(key, eqns, dual_env, invars, flat_duals):
            jax_util.safe_map(dual_env.write, invars, flat_duals)

            for eqn_idx, eqn in list(enumerate(eqns)):
                with src_util.user_context(eqn.source_info.traceback):
                    in_vals = jax_util.safe_map(dual_env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    duals = subfuns + in_vals

                    # Our sample_p primitive.
                    if eqn.primitive is sample_p:
                        dual_env = dual_env.copy()
                        pure_env = Dual.tree_primal(dual_env)

                        # Create pure continuation.
                        def _sample_pure_kont(key, *args):
                            return eval_jaxpr_iterate_pure(
                                key,
                                eqns[eqn_idx + 1 :],
                                pure_env,
                                eqn.outvars,
                                [*args],
                            )

                        # Create dual continuation.
                        def _sample_dual_kont(key, tree_dual):
                            dual_leaves = Dual.tree_leaves(tree_dual)
                            return eval_jaxpr_iterate_dual(
                                key,
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                dual_leaves,
                            )

                        in_tree = params["in_tree"]
                        num_consts = params["num_consts"]

                        flat_primals, flat_tangents = ADInterpreter.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals[num_consts:]))
                        )
                        adev_prim, *primals = jtu.tree_unflatten(in_tree, flat_primals)
                        _, *tangents = jtu.tree_unflatten(in_tree, flat_tangents)
                        tree_dual = Dual.tree_dual(primals, tangents)

                        return adev_prim.jvp_estimate(
                            key,
                            tree_dual,
                            (_sample_pure_kont, _sample_dual_kont),
                        )

                    # Handle branching.
                    elif eqn.primitive is jax.lax.cond_p:
                        pure_env = Dual.tree_primal(dual_env)

                        # Create dual continuation for the computation after the cond_p.
                        def _cond_dual_kont(tree_dual: List):
                            dual_leaves = Dual.tree_pure(tree_dual)
                            return eval_jaxpr_iterate_dual(
                                key,
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                dual_leaves,
                            )

                        branch_adev_functions = list(
                            map(
                                lambda fn: ADInterpreter.forward_mode(
                                    jc.jaxpr_as_fun(fn),
                                    _cond_dual_kont,
                                ),
                                params["branches"],
                            )
                        )

                        # NOTE: the branches are stored in the params
                        # in reverse order, so we need to reverse them
                        # This could totally be something which breaks in the future...
                        return jax.lax.cond(
                            Dual.tree_primal(in_vals[0]),
                            *reversed(branch_adev_functions),
                            key,
                            in_vals[1:],
                        )

                    # Default JVP rule for other JAX primitives.
                    else:
                        flat_primals, flat_tangents = ADInterpreter.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals))
                        )
                        if len(flat_primals) == 0:
                            primal_outs = eqn.primitive.bind(*flat_primals, **params)
                            tangent_outs = jtu.tree_map(jnp.zeros_like, primal_outs)
                        else:
                            jvp = jax_autodiff.primitive_jvps.get(eqn.primitive)
                            if not jvp:
                                msg = f"differentiation rule for '{eqn.primitive}' not implemented"
                                raise NotImplementedError(msg)
                            primal_outs, tangent_outs = jvp(
                                flat_primals, flat_tangents, **params
                            )

                if not eqn.primitive.multiple_results:
                    primal_outs = [primal_outs]
                    tangent_outs = [tangent_outs]

                jax_util.safe_map(
                    dual_env.write,
                    eqn.outvars,
                    Dual.tree_dual(primal_outs, tangent_outs),
                )
            (out_dual,) = jax_util.safe_map(dual_env.read, jaxpr.outvars)
            if not isinstance(out_dual, Dual):
                out_dual = Dual(out_dual, jnp.zeros_like(out_dual))
            return out_dual

        return eval_jaxpr_iterate_dual(
            key, jaxpr.eqns, dual_env, jaxpr.invars, flat_duals
        )

    @staticmethod
    def forward_mode(f, kont=lambda v: v):
        def _inner(key, tree_dual: Pytree):
            primals = jtu.tree_leaves(Dual.tree_primal(tree_dual))
            closed_jaxpr, (_, _, out_tree) = stage(f)(*primals)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            dual_leaves = Dual.tree_leaves(Dual.tree_pure(tree_dual))
            out_duals = ADInterpreter._eval_jaxpr_adev_jvp(
                key,
                jaxpr,
                consts,
                dual_leaves,
            )
            out_tree_def = out_tree()
            tree_primals, tree_tangents = Dual.tree_unzip(out_duals)
            out_tree_dual = Dual.tree_dual(
                jtu.tree_unflatten(out_tree_def, tree_primals),
                jtu.tree_unflatten(out_tree_def, tree_tangents),
            )
            vs = kont(out_tree_dual)
            return vs

        # Force coercion to JAX arrays.
        def maybe_array(v):
            return jnp.array(v, copy=False)

        @typecheck
        def _dual(key, tree_dual: Any):
            tree_dual = jtu.tree_map(maybe_array, tree_dual)
            return _inner(key, tree_dual)

        return _dual


#################
# ADEV programs #
#################


@Pytree.dataclass
class ADEVProgram(Pytree):
    source: Callable = Pytree.static()

    @typecheck
    def _jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,  # Pytree with Dual leaves.
        dual_kont: Callable,
    ) -> Dual:
        def adev_jvp(f):
            @wraps(f)
            def wrapped(tree_dual: Pytree):
                return ADInterpreter.forward_mode(self.source, dual_kont)(
                    key, tree_dual
                )

            return wrapped

        return adev_jvp(self.source)(tree_dual)

    def _jvp_estimate_identity_kont(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
    ):
        # Trivial continuation.
        def _identity(x):
            return x

        return self._jvp_estimate(key, primals, tangents, _identity)


###############
# Expectation #
###############


@Pytree.dataclass
class Expectation(Pytree):
    prog: ADEVProgram

    def jvp_estimate(self, key: PRNGKey, tree_dual: Pytree):
        # Trivial continuation.
        def _identity(v):
            return v

        return self.prog._jvp_estimate(key, tree_dual, _identity)

    def estimate(self, key, args):
        tangents = jtu.tree_map(lambda _: 0.0, args)
        primal, _ = self.jvp_estimate(key, args, tangents)
        return primal

    ##################################
    # JAX's native `grad` interface. #
    ##################################

    # The JVP rules here are registered below.
    # (c.f. Register custom forward mode with JAX)
    def grad_estimate(self, key: PRNGKey, primals: Tuple):
        def _invoke_closed_over(primals):
            return invoke_closed_over(self, key, primals)

        return jax.grad(_invoke_closed_over)(primals)

    def value_and_grad_estimate(self, key: PRNGKey, primals: Tuple):
        def _invoke_closed_over(primals):
            return invoke_closed_over(self, key, primals)

        return jax.value_and_grad(_invoke_closed_over)(primals)

    #################
    # For debugging #
    #################

    @typecheck
    def debug_transform_adev(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
    ):
        def _identity(x):
            return x

        return self.prog.debug_transform_adev(key, primals, tangents, _identity)


@typecheck
def expectation(source: Callable):
    prog = ADEVProgram(source)
    return Expectation(prog)


#########################################
# Register custom forward mode with JAX #
#########################################


# These two functions are defined to external to `Expectation`
# to ignore complexities with defining custom JVP rules for Pytree classes.
@jax.custom_jvp
def invoke_closed_over(instance, key, args):
    return instance.estimate(key, args)


def invoke_closed_over_jvp(primals, tangents):
    (instance, key, primals) = primals
    (_, _, tangents) = tangents
    duals = Dual.tree_dual(primals, tangents)
    out_dual = instance.jvp_estimate(key, duals)
    (v,), (tangent,) = Dual.tree_unzip(out_dual)
    return v, tangent


invoke_closed_over.defjvp(invoke_closed_over_jvp, symbolic_zeros=False)

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
from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import core as jc
from jax import util as jax_util
from jax.extend import source_info_util as src_util
from jax.interpreters import ad as jax_autodiff
from jax.interpreters.ad import Zero, instantiate_zeros

from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
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
        primals: Pytree,
        tangents: Pytree,
        konts: Tuple,
    ) -> Tuple:
        pass

    @typecheck
    def __call__(self, *args):
        key = reap_key()
        return sample_primitive(self, key, *args)


####################
# Sample intrinsic #
####################


sample_p = InitialStylePrimitive("sample")


@typecheck
def sample_primitive(adev_prim: ADEVPrimitive, key, *args):
    def _abstract_adev_prim_call(adev_prim, key, *args):
        v = adev_prim.sample(key, *args)
        return v

    return initial_style_bind(sample_p)(_abstract_adev_prim_call)(
        adev_prim,
        key,
        *args,
    )


#########################
# Reaping key intrinsic #
#########################

# Must be intercepted by e.g. an interpreter if we want randomness.
reap_key_p = InitialStylePrimitive("reap_key")


def reap_key():
    """
    The `reap_key` intrinsic inserts a primitive into a JAX computation that can be seeded with a fresh key by an interpreter.

    It should only be used within ADEV programs (as the ADEV transformation stack will
    handle sowing keys).
    """

    def _reap_key():
        # value doesn't matter, just the type
        return jax.random.PRNGKey(0)

    return initial_style_bind(reap_key_p)(_reap_key)()


####################
# ADEV interpreter #
####################


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

    # TODO: handle `jax.lax.cond`.
    @staticmethod
    def _eval_jaxpr_reap_key(key, jaxpr, consts, flat_args):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.invars, flat_args)
        jax_util.safe_map(env.write, jaxpr.constvars, consts)

        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            if eqn.primitive == reap_key_p:
                key, sub_key = jax.random.split(key)
                outvals = [sub_key]
            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)
        return jax_util.safe_map(env.read, jaxpr.outvars)

    @staticmethod
    def sow_keys(fn):
        """Return a transformed function which accepts a `key: PRNGKey` as the first argument, and uses an interpreter to sow fresh keys (using the `key`) into any `reap_key_p` primitive invocations.

        When a fresh key is sown, the carried key is split and evolved forward.
        """

        @wraps(fn)
        def wrapped(key, *args):
            closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            flat_out = ADInterpreter._eval_jaxpr_reap_key(key, jaxpr, consts, flat_args)
            return jtu.tree_unflatten(out_tree(), flat_out)

        return wrapped

    @staticmethod
    def _eval_jaxpr_adev_jvp(
        jaxpr: jc.Jaxpr,
        consts: List,
        flat_duals: List,
    ):
        dual_env = Environment()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, Dual.tree_pure(consts))
        jax_util.safe_map(dual_env.write, jaxpr.invars, flat_duals)

        def eval_jaxpr_iterate(eqns, dual_env, invars, flat_duals):
            jax_util.safe_map(dual_env.write, invars, flat_duals)

            for eqn_idx, eqn in list(enumerate(eqns)):
                with src_util.user_context(eqn.source_info.traceback):
                    in_vals = jax_util.safe_map(dual_env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    duals = subfuns + in_vals

                    if eqn.primitive is sample_p:
                        dual_env = dual_env.copy()
                        pure_env = Dual.tree_primal(dual_env)

                        # Create pure continuation.
                        def pure_kont(*args):
                            return eval_jaxpr_iterate(
                                eqns[eqn_idx + 1 :], pure_env, eqn.outvars, [*args]
                            )

                        # Create dual continuation.
                        def dual_kont(primals, tangents):
                            duals = Dual.tree_dual(primals, tangents)
                            out_dual = eval_jaxpr_iterate(
                                eqns[eqn_idx + 1 :], dual_env, eqn.outvars, [*duals]
                            )
                            if isinstance(out_dual, Dual):
                                return out_dual.primal, instantiate_zeros(
                                    out_dual.tangent
                                )
                            else:
                                return out_dual, 0.0

                        in_tree = params["in_tree"]
                        num_consts = params["num_consts"]

                        flat_primals, flat_tangents = ADInterpreter.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals[num_consts:]))
                        )
                        adev_prim, key, *primals = jtu.tree_unflatten(
                            in_tree, flat_primals
                        )
                        _, _, *tangents = jtu.tree_unflatten(in_tree, flat_tangents)

                        primal_out, tangent_out = adev_prim.jvp_estimate(
                            key,
                            primals,
                            tangents,
                            (pure_kont, dual_kont),
                        )
                        return Dual(primal_out, tangent_out)

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
            return out_dual

        return eval_jaxpr_iterate(jaxpr.eqns, dual_env, jaxpr.invars, flat_duals)

    @staticmethod
    def forward_mode(f, kont=lambda v: v):
        def _konted(*args):
            return kont(f(*args))

        def _inner(primals: Tuple, tangents: Tuple):
            closed_jaxpr, (flat_args, _, _) = stage(_konted)(*primals)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            flat_tangents = jtu.tree_leaves(
                tangents, is_leaf=lambda v: isinstance(v, Zero)
            )
            out_dual = ADInterpreter._eval_jaxpr_adev_jvp(
                jaxpr,
                consts,
                Dual.tree_dual(flat_args, flat_tangents),
            )
            if isinstance(out_dual, Dual):
                return out_dual.primal, instantiate_zeros(out_dual.tangent)
            else:
                return out_dual, 0.0

        # Force coercion to JAX arrays.
        def maybe_array(v):
            return jnp.array(v, copy=False)

        @typecheck
        def _dual(primals: Tuple, tangents: Tuple):
            primals = jtu.tree_map(maybe_array, primals)
            tangents = jtu.tree_map(maybe_array, tangents)
            return _inner(primals, tangents)

        return _dual


#################
# ADEV programs #
#################


class ADEVProgram(Pytree):
    source: Callable = Pytree.static()

    @typecheck
    def _jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        kont: Callable,
    ):
        def adev_jvp(f):
            @wraps(f)
            def wrapped(primals, tangents):
                sown = partial(ADInterpreter.sow_keys(f), key)
                return ADInterpreter.forward_mode(sown, kont)(primals, tangents)

            return wrapped

        return adev_jvp(self.source)(primals, tangents)

    def _jvp_estimate_identity_kont(self, key, primals, tangents):
        # Trivial continuation.
        def _identity(x):
            return x

        return self._jvp_estimate(key, primals, tangents, _identity)

    #################
    # For debugging #
    #################

    @typecheck
    def debug_transform_sow(
        self,
        key: PRNGKey,
        args: Tuple,
    ):
        def sown(f):
            @wraps(f)
            def _sown(*args):
                sown = partial(ADInterpreter.sow_keys(f), key)
                return sown(*args)

            return _sown

        value = sown(self.source)(*args)
        jaxpr = jax.make_jaxpr(sown(self.source))(*args)
        return value, jaxpr

    @typecheck
    def debug_transform_adev(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        kont: Callable,
    ):
        def adev_jvp(f):
            @wraps(f)
            def wrapped(primals, tangents):
                sown = partial(ADInterpreter.sow_keys(f), key)
                return ADInterpreter.forward_mode(sown, kont)(primals, tangents)

            return wrapped

        value = adev_jvp(self.source)(primals, tangents)
        jaxpr = jax.make_jaxpr(adev_jvp(self.source))(primals, tangents)
        return value, jaxpr


###############
# Expectation #
###############


class Expectation(Pytree):
    prog: ADEVProgram

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple[Pytree, ...],
        tangents: Tuple[Pytree, ...],
    ):
        # Trivial continuation.
        def _identity(v):
            return v

        return self.prog._jvp_estimate(key, primals, tangents, _identity)

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
    def debug_transform_sow(
        self,
        key: PRNGKey,
        args: Tuple,
    ):
        def _identity(x):
            return x

        return self.prog.debug_transform_sow(key, args, _identity)

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
    v, tangent = instance.jvp_estimate(key, primals, tangents)
    return v, tangent


invoke_closed_over.defjvp(invoke_closed_over_jvp, symbolic_zeros=False)

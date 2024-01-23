# Copyright 2022 MIT Probabilistic Computing Project
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
from jax import api_util
from jax import util as jax_util
from jax._src import core
from jax._src.interpreters import partial_eval as pe
from jax.extend import linear_util as lu
from jax.interpreters import ad as jax_autodiff
from jax.interpreters.ad import Zero, instantiate_zeros

from genjax._src.core.interpreters.cps import cps
from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Int,
    PRNGKey,
    Sequence,
    Tuple,
    Union,
    dispatch,
    typecheck,
)

###################
# ADEV primitives #
###################


class ADEVPrimitive(Pytree):
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
        return sample(self, *args)


####################
# Sample intrinsic #
####################


sample_p = InitialStylePrimitive("sample")


@typecheck
def sample_with_key(adev_prim: ADEVPrimitive, key, *args):
    def _abstract_adev_prim_call(adev_prim, key, *args):
        v = adev_prim.sample(key, *args)
        return v

    return initial_style_bind(sample_p)(_abstract_adev_prim_call)(
        adev_prim,
        key,
        *args,
    )


@typecheck
def sample(adev_prim: ADEVPrimitive, *args):
    key = reap_key()
    return sample_with_key(adev_prim, key, *args)


######################
# Seeding randomness #
######################

seed_p = InitialStylePrimitive("reap_key")


def reap_key():
    def _inner():
        # Only the type and shape matter.
        return jax.random.PRNGKey(0)

    return initial_style_bind(seed_p)(_inner)()


def _reap_key_jaxpr(key, jaxpr, consts, flat_args):
    env = Environment()
    jax_util.safe_map(env.write, jaxpr.invars, flat_args)
    jax_util.safe_map(env.write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = jax_util.safe_map(env.read, eqn.invars)
        subfuns, params = eqn.primitive.get_bind_params(eqn.params)
        args = subfuns + invals
        if eqn.primitive == seed_p:
            key, sub_key = jax.random.split(key)
            outvals = [sub_key]
        else:
            outvals = eqn.primitive.bind(*args, **params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        jax_util.safe_map(env.write, eqn.outvars, outvals)
    return jax_util.safe_map(env.read, jaxpr.outvars)


def sow_key(fn):
    def wrapped(key, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        out = _reap_key_jaxpr(key, jaxpr, consts, flat_args)
        return jtu.tree_unflatten(out_tree(), out)

    return wrapped


####################
# ADEV interpreter #
####################


class Dual(Pytree):
    primal: Any
    tangent: Any

    @classmethod
    def pure(cls, v):
        return Dual(v, jnp.zeros_like(v))


def tree_dual_pure(c):
    return jtu.tree_map(lambda v: Dual(v, jnp.zeros_like(v)), c)


def maybe_dual(c):
    def _inner(v):
        if isinstance(v, Dual):
            return v
        else:
            return Dual.pure(v)

    return list(map(_inner, c))


def tree_dual(primals, tangents):
    return jtu.tree_map(lambda v1, v2: Dual(v1, v2), primals, tangents)


def flat_unzip(duals):
    primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in duals)
    return list(primals), list(tangents)


def tree_dual_primal(v):
    def _inner(v):
        if isinstance(v, Dual):
            return v.primal
        else:
            return v

    return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))


def tree_dual_tangent(v):
    def _inner(v):
        if isinstance(v, Dual):
            return v.tangent
        else:
            return v

    return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))


class PytreeContinuationClosure(Pytree):
    dual_env: Environment
    callable: Callable = Pytree.static()

    def pure(self, *args):
        just_primal_env = tree_dual_primal(self.dual_env)
        (out_retval,) = self.callable(just_primal_env, *args)
        return out_retval

    @dispatch
    def dual(self, primals: Tuple, tangents: Tuple):
        just_primal_env = tree_dual_primal(self.dual_env)
        just_tangent_env = tree_dual_tangent(self.dual_env)
        return dual(self.callable)(
            (just_primal_env, *primals),
            (just_tangent_env, *tangents),
        )

    @dispatch
    def dual(self, primals: Tuple):
        just_primal_env = tree_dual_primal(self.dual_env)
        just_tangent_env = tree_dual_tangent(self.dual_env)
        return dual(self.callable)(
            (just_primal_env, *primals),
            (just_tangent_env, *jtu.tree_map(jnp.zeros_like, primals)),
        )

    def __call__(self, *args):
        return self.pure(*args)


def pytree_continuation_closure(callable, dual_env):
    return PytreeContinuationClosure(dual_env, callable)


def _eval_jaxpr_adev_jvp(jaxpr, consts, flat_duals):
    dual_env = Environment()
    jax_util.safe_map(dual_env.write, jaxpr.constvars, tree_dual_pure(consts))
    jax_util.safe_map(dual_env.write, jaxpr.invars, flat_duals)

    for eqn in jaxpr.eqns:
        duals = jax_util.safe_map(dual_env.read, eqn.invars)
        _, params = eqn.primitive.get_bind_params(eqn.params)

        if eqn.primitive is sample_p:
            num_consts = params["num_consts"]
            kont = params["kont"]
            in_tree = params["in_tree"]
            flat_primals, flat_tangents = flat_unzip(maybe_dual(duals[num_consts:]))
            args = flat_primals
            _, (adev_prim, key, *primals) = jtu.tree_unflatten(in_tree, args)
            _, (_, _, *tangents) = jtu.tree_unflatten(in_tree, flat_tangents)
            kont_closure = pytree_continuation_closure(kont, dual_env)
            primal_out, tangent_out = adev_prim.jvp_estimate(
                key,
                primals,
                tangents,
                (kont_closure.pure, kont_closure.dual),
            )
            return Dual(primal_out, tangent_out)

        else:
            flat_primals, flat_tangents = flat_unzip(maybe_dual(duals))
            if len(flat_primals) == 0:
                primal_outs = eqn.primitive.bind(*flat_primals, **params)
                tangent_outs = jnp.zeros_like(primal_outs)
            else:
                jvp = jax_autodiff.primitive_jvps.get(eqn.primitive)
                if not jvp:
                    msg = f"Differentiation rule for '{eqn.primitive}' not implemented"
                    raise NotImplementedError(msg)
                primal_outs, tangent_outs = jvp(flat_primals, flat_tangents, **params)

        if not eqn.primitive.multiple_results:
            primal_outs = [primal_outs]
            tangent_outs = [tangent_outs]

        jax_util.safe_map(
            dual_env.write,
            eqn.outvars,
            tree_dual(primal_outs, tangent_outs),
        )

    (out_dual,) = jax_util.safe_map(dual_env.read, jaxpr.outvars)
    return out_dual


def maybe_array(v):
    return jnp.array(v, copy=False)


def dual(f):
    def _inner(primals: Tuple, tangents: Tuple):
        closed_jaxpr, (flat_args, _, _) = stage(f)(*primals)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_tangents = jtu.tree_leaves(tangents, is_leaf=lambda v: isinstance(v, Zero))
        out_dual = _eval_jaxpr_adev_jvp(
            jaxpr, consts, tree_dual(flat_args, flat_tangents)
        )
        if isinstance(out_dual, Dual):
            return out_dual.primal, instantiate_zeros(out_dual.tangent)
        else:
            return out_dual, 0.0

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

    # For debugging.
    @typecheck
    def debug_transform_seed(
        self,
        key: PRNGKey,
        args: Tuple,
    ):
        def _just_seed(f):
            @wraps(f)
            def wrapped(key, args):
                def _seeded(*args):
                    return sow_key(f)(key, *args)

                return _seeded(*args)

            return wrapped

        return _just_seed(self.source)(key, args), jax.make_jaxpr(
            _just_seed(self.source)
        )(key, args)

    # For debugging.
    @typecheck
    def debug_transform_cps(
        self,
        key: PRNGKey,
        args: Tuple,
        kont: Callable,
    ):
        def _just_cps(f):
            @wraps(f)
            def wrapped(key, args):
                def _seeded(*args):
                    return sow_key(f)(key, *args)

                def cpsified(*args):
                    return cps(_seeded, kont, [sample_p])(*args)

                return cpsified(*args)

            return wrapped

        return _just_cps(self.source)(key, args), jax.make_jaxpr(
            _just_cps(self.source)
        )(key, args)

    # For debugging.
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
            def wrapped(key, primals, tangents):
                def _seeded(*args):
                    return sow_key(f)(key, *args)

                def cpsified(*args):
                    return cps(_seeded, kont, [sample_p])(*args)

                return dual(cpsified)(primals, tangents)

            return wrapped

        return adev_jvp(self.source)(key, primals, tangents), jax.make_jaxpr(
            adev_jvp(self.source)
        )(key, primals, tangents)

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
            def wrapped(key, primals, tangents):
                def _seeded(*args):
                    return sow_key(f)(key, *args)

                def cpsified(*args):
                    return cps(_seeded, kont, [sample_p])(*args)

                return dual(cpsified)(primals, tangents)

            return wrapped

        return adev_jvp(self.source)(key, primals, tangents)

    def _jvp_estimate_identity_kont(self, key, primals, tangents):
        # Trivial continuation.
        def _identity(x):
            return x

        return self._jvp_estimate(key, primals, tangents, _identity)

    # For debugging -- `grad_estimate` just uses `jax.grad`
    # to automatically do the below.
    def _grad_estimate(self, key, args):
        @typecheck
        def _inner(key: PRNGKey, args: Tuple):
            # Force to arrayful, to support getting shapes and dtypes.
            args = jtu.tree_map(maybe_array, args)
            primal_tree = jtu.tree_structure(args)
            # Second args slot is for tangents.
            _, in_tree = jtu.tree_flatten((key, args, args))
            fun = lu.wrap_init(self._jvp_estimate_identity_kont)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
            # Flat known is the key and the arguments.
            flat_known = jtu.tree_map(
                pe.PartialVal.known,
                jtu.tree_leaves((key, args)),
            )

            # Flat unknown is the tangents.
            flat_unknown = jtu.tree_map(
                lambda v: pe.PartialVal.unknown(core.ShapedArray(v.shape, v.dtype)),
                jtu.tree_leaves(args),
            )
            in_pvals = [*flat_known, *flat_unknown]
            jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(flat_fun, in_pvals)
            primal_dummies = [
                jax_autodiff.UndefinedPrimal(v.aval) for v in jaxpr.invars
            ]
            flat_args_bar = jax_autodiff.backward_pass(
                jaxpr, (), None, consts, primal_dummies, (1.0,)
            )
            args_bar = jtu.tree_unflatten(primal_tree, flat_args_bar)
            return args_bar

        return _inner(key, args)


@typecheck
def adev(callable: Callable):
    return ADEVProgram(callable)


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

    @dispatch
    def grad_estimate(self, key: PRNGKey, primals: Tuple):
        def _invoke_closed_over(primals):
            return invoke_closed_over(self, key, primals)

        return jax.grad(_invoke_closed_over)(primals)

    # Can be used to customize how `jax.grad` is applied
    # -- e.g. by specifying which arguments the gradient should be taken
    # with respect to.
    @dispatch
    def grad_estimate(self, argnums: Union[Int, Sequence[Int]] = 0):
        def _invoke_closed_over(key, *primals):
            return invoke_closed_over(self, key, primals)

        @typecheck
        def _grad(key: PRNGKey, *primals: Any):
            return jax.grad(
                partial(_invoke_closed_over, key),
                argnums=argnums,
            )(*primals)

        return _grad

    @dispatch
    def value_and_grad_estimate(self, key: PRNGKey, primals: Tuple):
        def _invoke_closed_over(primals):
            return invoke_closed_over(self, key, primals)

        return jax.value_and_grad(_invoke_closed_over)(primals)

    @dispatch
    def value_and_grad_estimate(self, argnums: Union[Int, Sequence[Int]]):
        def _invoke_closed_over(key, *primals):
            return invoke_closed_over(self, key, primals)

        @typecheck
        def _grad(key: PRNGKey, *primals: Any):
            return jax.value_and_grad(
                partial(_invoke_closed_over, key),
                argnums=argnums,
            )(*primals)

        return _grad

    # For debugging.
    def grad_estimate_custom(self, key, primals):
        return self.prog._grad_estimate(key, primals)

    def __call__(self, key, *args):
        return invoke_closed_over(self, key, args)


# These two functions are defined to ignore complexities
# with defining custom JVP rules for Pytree classes.
@jax.custom_jvp
def invoke_closed_over(instance, key, args):
    return instance.estimate(key, args)


def invoke_closed_over_jvp(primals, tangents):
    (instance, key, primals) = primals
    (_, _, tangents) = tangents
    v, tangent = instance.jvp_estimate(key, primals, tangents)
    return v, tangent


invoke_closed_over.defjvp(invoke_closed_over_jvp, symbolic_zeros=False)


@typecheck
def E(prog: ADEVProgram):
    return Expectation(prog)

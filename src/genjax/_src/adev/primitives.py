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
"""Defines ADEV primitives."""

import jax
import jax._src.core
import jax._src.dtypes as jax_dtypes
import jax.numpy as jnp
from jax._src.ad_util import Zero
from jax._src.core import (
    get_aval,
    raise_to_shaped,
)
from jax.interpreters.ad import zeros_like_aval
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    Dual,
    DualTree,
    TailCallADEVPrimitive,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    bernoulli,
    categorical,
    geometric,
    normal,
)

tfd = tfp.distributions

# These methods are pulled from jax.interpreters.ad:


def instantiate_zeros(tangent):
    return zeros_like_aval(tangent.aval) if type(tangent) is Zero else tangent


def zeros_like_jaxval(val):
    return zeros_like_aval(raise_to_shaped(get_aval(val)))


def recast_to_float0(primal, tangent):
    if (
        jax._src.core.primal_dtype_to_tangent_dtype(jax_dtypes.dtype(primal))
        == jax_dtypes.float0
    ):
        return Zero(jax._src.core.get_aval(primal).at_least_vspace())
    else:
        return tangent


def zero(v):
    ad_zero = recast_to_float0(v, zeros_like_jaxval(v))
    return instantiate_zeros(ad_zero)


################################
# Gradient strategy primitives #
################################


@Pytree.dataclass
class REINFORCE(ADEVPrimitive):
    sample_function: Callable[..., Any] = Pytree.static()
    differentiable_logpdf: Callable[..., Any] = Pytree.static()

    def sample(self, *args):
        return self.sample_function(*args)

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        primals = Dual.tree_primal(dual_tree)
        tangents = Dual.tree_tangent(dual_tree)
        v = self.sample(*primals)
        dual_tree = Dual.tree_pure(v)
        out_dual = kdual(dual_tree)
        (out_primal,), (out_tangent,) = Dual.tree_unzip(out_dual)
        _, lp_tangent = jax.jvp(
            self.differentiable_logpdf,
            (v, *primals),
            (zero(v), *tangents),
        )
        return Dual(out_primal, out_tangent + (out_primal * lp_tangent))


def reinforce(sample_func, logpdf_func):
    return REINFORCE(sample_func, logpdf_func)


###########################
# Distribution primitives #
###########################


@Pytree.dataclass
class FlipEnum(ADEVPrimitive):
    def sample(self, *args):
        (probs,) = args
        return 1 == bernoulli.sample(probs)

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        true_dual = kdual(
            Dual(jnp.array(True), jnp.zeros_like(jnp.array(True))),
        )
        false_dual = kdual(
            Dual(jnp.array(False), jnp.zeros_like(jnp.array(False))),
        )
        (true_primal,), (true_tangent,) = Dual.tree_unzip(true_dual)
        (false_primal,), (false_tangent,) = Dual.tree_unzip(false_dual)

        def _inner(p, tl, fl):
            return p * tl + (1 - p) * fl

        out_primal, out_tangent = jax.jvp(
            _inner,
            (p_primal, true_primal, false_primal),
            (p_tangent, true_tangent, false_tangent),
        )
        return Dual(out_primal, out_tangent)


flip_enum = FlipEnum()


@Pytree.dataclass
class FlipMVD(ADEVPrimitive):
    def sample(self, *args):
        p = (args,)
        return 1 == bernoulli.sample(p)

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (kpure, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_primal(dual_tree)
        v = bernoulli.sample(p_primal)
        b = v == 1
        b_primal, b_tangent = kdual((b,), (jnp.zeros_like(b),))
        other = kpure(jnp.logical_not(b))
        est = ((-1) ** v) * (other - b_primal)
        return Dual(b_primal, b_tangent + est * p_tangent)


flip_mvd = FlipMVD()


@Pytree.dataclass
class FlipEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (p,) = args
        return 1 == bernoulli.sample(p)

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        ret_primals, ret_tangents = jax.vmap(kdual)(
            (jnp.array([True, False]),),
            (jnp.zeros_like(jnp.array([True, False]))),
        )

        def _inner(p, ret):
            return jnp.sum(jnp.array([p, 1 - p]) * ret)

        return Dual(
            *jax.jvp(
                _inner,
                (p_primal, ret_primals),
                (p_tangent, ret_tangents),
            )
        )


flip_enum_parallel = FlipEnumParallel()


@Pytree.dataclass
class CategoricalEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (probs,) = args
        return categorical.sample(probs)

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (probs_primal,) = Dual.tree_primal(dual_tree)
        (probs_tangent,) = Dual.tree_tangent(dual_tree)
        idxs = jnp.arange(len(probs_primal))
        ret_primals, ret_tangents = jax.vmap(kdual)((idxs,), (jnp.zeros_like(idxs),))

        def _inner(probs, primals):
            return jnp.sum(jax.nn.softmax(probs) * primals)

        return Dual(
            *jax.jvp(
                _inner,
                (probs_primal, ret_primals),
                (probs_tangent, ret_tangents),
            )
        )


categorical_enum_parallel = CategoricalEnumParallel()

flip_reinforce = reinforce(
    lambda p: 1 == bernoulli.sample(p),
    lambda v, p: bernoulli.logpdf(v, p),
)

geometric_reinforce = reinforce(
    lambda args: geometric.sample(*args), lambda v, args: geometric.logpdf(v, *args)
)

normal_reinforce = reinforce(
    lambda loc, scale: normal.sample(loc, scale),
    lambda v, loc, scale: normal.logpdf(v, loc, scale),
)


@Pytree.dataclass
class NormalREPARAM(TailCallADEVPrimitive):
    def sample(self, *args):
        loc, scale_diag = args
        return normal.sample(loc, scale_diag)

    def before_tail_call(
        self,
        dual_tree: DualTree,
    ) -> Dual:
        (mu_primal, sigma_primal) = Dual.tree_primal(dual_tree)
        (mu_tangent, sigma_tangent) = Dual.tree_tangent(dual_tree)
        eps = normal.sample(0.0, 1.0)

        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return Dual(primal_out, tangent_out)


normal_reparam = NormalREPARAM()


@Pytree.dataclass
class Uniform(TailCallADEVPrimitive):
    def sample(self, *_args):
        return uniform.sample(0.0, 1.0)

    def before_tail_call(
        self,
        dual_tree: tuple[Any, ...],
    ):
        x = uniform.sample(0.0, 1.0)
        return Dual(x, 0.0)


uniform = Uniform()


@Pytree.dataclass
class BetaIMPLICIT(TailCallADEVPrimitive):
    def sample(self, *args):
        alpha, beta = args
        return beta.sample(alpha, beta)

    def before_tail_call(
        self,
        dual_tree: DualTree,
    ):
        # Because TFP already overloads their Beta sampler with implicit
        # differentiation rules for JVP, we directly utilize their rules.
        def _inner(alpha, beta):
            # Invoking TFP's Implicit reparametrization:
            # https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/distributions/beta.py#L292-L306
            x = beta.sample(alpha, beta)
            return x

        # We invoke JAX's JVP (which utilizes TFP's registered implicit differentiation
        # rule for Beta) to get a primal and tangent out.
        primals = Dual.tree_primal(dual_tree)
        tangents = Dual.tree_tangent(dual_tree)
        primal_out, tangent_out = jax.jvp(_inner, primals, tangents)
        return Dual(primal_out, tangent_out)


beta_implicit = BetaIMPLICIT()


@Pytree.dataclass
class Baseline(ADEVPrimitive):
    prim: ADEVPrimitive

    def sample(self, *args):
        return self.prim.sample(*args[1:])

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (kpure, kdual) = konts
        (b_primal, *prim_primals) = Dual.tree_primal(dual_tree)
        (b_tangent, *prim_tangents) = Dual.tree_tangent(dual_tree)

        def new_kdual(dual: Dual):
            ret_dual = kdual(dual)

            def _inner(ret, b):
                return ret - b

            primal, tangent = jax.jvp(
                _inner,
                (ret_dual.primal, b_primal),
                (ret_dual.tangent, b_tangent),
            )
            return Dual(primal, tangent)

        l_dual = self.prim.jvp_estimate(
            Dual.dual_tree(prim_primals, prim_tangents),
            (kpure, new_kdual),
        )

        def _inner(left, right):
            return left + right

        primal, tangent = jax.jvp(
            _inner,
            (l_dual.primal, b_primal),
            (l_dual.tangent, b_tangent),
        )
        return Dual(primal, tangent)


def baseline(prim):
    return Baseline(prim)


##################
# Loss primitive #
##################


@Pytree.dataclass
class AddCost(ADEVPrimitive):
    def sample(self, *args):
        (w,) = args
        return w

    def jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ) -> Dual:
        (_, kdual) = konts
        (w,) = Dual.tree_primal(dual_tree)
        (w_tangent,) = Dual.tree_tangent(dual_tree)
        l_dual = kdual(Dual(None, None))
        return Dual(w + l_dual.primal, w_tangent + l_dual.tangent)


def add_cost(w):
    prim = AddCost()
    prim(w)

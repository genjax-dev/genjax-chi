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
import jax.numpy as jnp
from jax.interpreters.ad import instantiate_zeros, recast_to_float0, zeros_like_jaxval
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    Dual,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    PRNGKey,
    Tuple,
    typecheck,
)

tfd = tfp.distributions


def zero(v):
    ad_zero = recast_to_float0(v, zeros_like_jaxval(v))
    return instantiate_zeros(ad_zero)


################################
# Gradient strategy primitives #
################################


@Pytree.dataclass
class REINFORCE(ADEVPrimitive):
    sample_function: Callable = Pytree.static()
    differentiable_logpdf: Callable = Pytree.static()

    def sample(self, key, *args):
        return self.sample_function(key, *args)

    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,
        konts: Tuple,
    ):
        (_, kdual) = konts
        primals = Dual.tree_primal(tree_dual)
        tangents = Dual.tree_tangent(tree_dual)
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *primals)
        tree_dual = Dual.tree_pure(v)
        out_dual = kdual(key, tree_dual)
        (out_primal,), (out_tangent,) = Dual.tree_unzip(out_dual)
        _, lp_tangent = jax.jvp(
            self.differentiable_logpdf,
            (v, *primals),
            (zero(v), *tangents),
        )
        return Dual(out_primal, out_tangent + (out_primal * lp_tangent))


@typecheck
def reinforce(sample_func, logpdf_func):
    return REINFORCE(sample_func, logpdf_func)


###########################
# Distribution primitives #
###########################


@Pytree.dataclass
class FlipEnum(ADEVPrimitive):
    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,
        konts: Tuple,
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(tree_dual)
        (p_tangent,) = Dual.tree_tangent(tree_dual)
        true_dual = kdual(
            key,
            Dual(jnp.array(True), jnp.zeros_like(jnp.array(True))),
        )
        false_dual = kdual(
            key,
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
class BernoulliMVD(ADEVPrimitive):
    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (p_primal,) = primals
        (p_tangent,) = tangents
        key, sub_key = jax.random.split(key)
        v = tfd.Bernoulli(probs=p_primal).sample(seed=sub_key)
        b = v == 1
        b_primal, b_tangent = kdual(key, (b,), (jnp.zeros_like(b),))
        other = kpure(key, jnp.logical_not(b))
        est = ((-1) ** v) * (other - b_primal)
        return b_primal, b_tangent + est * p_tangent


flip_mvd = BernoulliMVD()


@Pytree.dataclass
class FlipEnumParallel(ADEVPrimitive):
    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (p_primal,) = primals
        (p_tangent,) = tangents
        sub_keys = jax.random.split(key, 2)
        ret_primals, ret_tangents = jax.vmap(kdual)(
            sub_keys,
            (jnp.array([True, False]),),
            (jnp.zeros_like(jnp.array([True, False]))),
        )

        def _inner(p, ret):
            return jnp.sum(jnp.array([p, 1 - p]) * ret)

        return jax.jvp(
            _inner,
            (p_primal, ret_primals),
            (p_tangent, ret_tangents),
        )


flip_enum_parallel = FlipEnumParallel()


@Pytree.dataclass
class CategoricalEnumParallel(ADEVPrimitive):
    def sample(self, key, probs):
        return tfd.Categorical(probs=probs).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (probs_primal,) = primals
        (probs_tangent,) = tangents
        idxs = jnp.arange(len(probs_primal))
        sub_keys = jax.random.split(key, len(probs_primal))
        ret_primals, ret_tangents = jax.vmap(kdual)(
            sub_keys, (idxs,), (jnp.zeros_like(idxs),)
        )

        def _inner(probs, primals):
            return jnp.sum(jax.nn.softmax(probs) * primals)

        return jax.jvp(
            _inner,
            (probs_primal, ret_primals),
            (probs_tangent, ret_tangents),
        )


categorical_enum_parallel = CategoricalEnumParallel()

flip_reinforce = reinforce(
    lambda key, p: 1 == tfd.Bernoulli(probs=p).sample(seed=key),
    lambda v, p: tfd.Bernoulli(probs=p).log_prob(v),
)

geometric_reinforce = reinforce(
    lambda key, args: tfd.Geometric(*args).sample(seed=key),
    lambda v, args: tfd.Geometric(*args).log_prob(v),
)

normal_reinforce = reinforce(
    lambda key, loc, scale: tfd.Normal(loc, scale).sample(seed=key),
    lambda v, loc, scale: tfd.Normal(loc, scale).log_prob(v),
)


@Pytree.dataclass
class NormalREPARAM(ADEVPrimitive):
    def sample(self, key, loc, scale_diag):
        return tfd.Normal(loc=loc, scale=scale_diag).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (mu_primal, sigma_primal) = Dual.tree_primal(tree_dual)
        (mu_tangent, sigma_tangent) = Dual.tree_tangent(tree_dual)
        key, sub_key = jax.random.split(key)
        eps = tfd.Normal(loc=0.0, scale=1.0).sample(seed=sub_key)

        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return kdual(key, Dual(primal_out, tangent_out))


normal_reparam = NormalREPARAM()


@Pytree.dataclass
class MvNormalDiagREPARAM(ADEVPrimitive):
    def sample(self, key, loc, scale_diag):
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag).sample(
            seed=key
        )

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (loc_primal, diag_scale_primal) = primals
        (loc_tangent, diag_scale_tangent) = tangents
        key, sub_key = jax.random.split(key)

        eps = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=loc_primal.shape, seed=sub_key
        )

        # This takes N samples from N(0.0, 1.0) and transforms
        # them to MvNormalDiag(loc, diag_scale).
        def _inner(loc, diag_scale):
            return loc + jnp.multiply(diag_scale, eps)

        primal_out, tangent_out = jax.jvp(
            _inner,
            (loc_primal, diag_scale_primal),
            (loc_tangent, diag_scale_tangent),
        )

        return kdual(key, (primal_out,), (tangent_out,))


mv_normal_diag_reparam = MvNormalDiagREPARAM()


@Pytree.dataclass
class MvNormalREPARAM(ADEVPrimitive):
    def sample(self, key, mu, sigma):
        v = tfd.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=sigma
        ).sample(seed=key)
        return v

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (mu_primal, cov_primal) = primals
        (mu_tangent, cov_tangent) = tangents
        key, sub_key = jax.random.split(key)

        eps = tfd.Normal(loc=0.0, scale=1.0).sample(len(mu_primal), seed=sub_key)

        def _inner(eps, mu, cov):
            L = jnp.linalg.cholesky(cov)
            return mu + L @ eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (eps, mu_primal, cov_primal),
            (jnp.zeros_like(eps), mu_tangent, cov_tangent),
        )
        return kdual(key, (primal_out,), (tangent_out,))


mv_normal_reparam = MvNormalREPARAM()


@Pytree.dataclass
class Uniform(ADEVPrimitive):
    def sample(
        self,
        key: PRNGKey,
    ):
        v = tfd.Uniform(low=0.0, high=1.0).sample(seed=key)
        return v

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        key, sub_key = jax.random.split(key)
        x = tfd.Uniform(low=0.0, high=1.0).sample(seed=sub_key)
        return kdual(key, (x,), (0.0,))


uniform = Uniform()


@Pytree.dataclass
class Baseline(ADEVPrimitive):
    prim: ADEVPrimitive

    def sample(self, key, b, *args):
        return self.prim.sample(key, *args)

    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (b_primal, *prim_primals) = Dual.tree_primal(tree_dual)
        (b_tangent, *prim_tangents) = Dual.tree_tangent(tree_dual)

        def new_kdual(key, dual: Dual):
            ret_dual = kdual(key, dual)

            def _inner(ret, b):
                return ret - b

            primal, tangent = jax.jvp(
                _inner,
                (ret_dual.primal, b_primal),
                (ret_dual.tangent, b_tangent),
            )
            return Dual(primal, tangent)

        l_dual = self.prim.jvp_estimate(
            key,
            Dual.tree_dual(prim_primals, prim_tangents),
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


@typecheck
def baseline(prim):
    return Baseline(prim)


##################
# Loss primitive #
##################


@Pytree.dataclass
class AddCost(ADEVPrimitive):
    def sample(self, key, w):
        return w

    def jvp_estimate(
        self,
        key: PRNGKey,
        tree_dual: Any,
        konts: Tuple,
    ) -> Dual:
        (kpure, kdual) = konts
        (w,) = Dual.tree_primal(tree_dual)
        (w_tangent,) = Dual.tree_tangent(tree_dual)
        l_dual = kdual(key, Dual(None, None))
        return Dual(w + l_dual.primal, w_tangent + l_dual.tangent)


def add_cost(w):
    prim = AddCost()
    prim(w)

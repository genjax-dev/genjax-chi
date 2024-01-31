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
"""Defines ADEV primitives."""


import jax
import jax.numpy as jnp
from jax.interpreters.ad import instantiate_zeros, recast_to_float0, zeros_like_jaxval
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
)
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
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


class REINFORCE(ADEVPrimitive):
    sample_function: Callable = Pytree.static()
    differentiable_logpdf: Callable = Pytree.static()

    def sample(self, key, *args):
        return self.sample_function(key, *args)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (_, kdual) = konts
        v = self.sample(key, *primals)
        l_primal, l_tangent = kdual((v,), (jnp.zeros_like(v),))
        _, lp_tangent = jax.jvp(
            self.differentiable_logpdf,
            (v, *primals),
            (zero(v), *tangents),
        )
        return l_primal, l_tangent + (l_primal * lp_tangent)


@typecheck
def reinforce(sample_func, logpdf_func):
    return REINFORCE(sample_func, logpdf_func)


###########################
# Distribution primitives #
###########################


class BernoulliEnum(ADEVPrimitive):
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
        tl_primal, tl_tangent = kdual(
            (jnp.array(True),),
            (jnp.zeros_like(jnp.array(True)),),
        )
        fl_primal, fl_tangent = kdual(
            (jnp.array(False),),
            (jnp.zeros_like(jnp.array(False)),),
        )

        def _inner(p, tl, fl):
            return p * tl + (1 - p) * fl

        return jax.jvp(
            _inner,
            (p_primal, tl_primal, fl_primal),
            (p_tangent, tl_tangent, fl_tangent),
        )


flip_enum = BernoulliEnum()


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
        v = tfd.Bernoulli(probs=p_primal).sample(seed=key)
        b = v == 1
        b_primal, b_tangent = kdual((b,), (jnp.zeros_like(b),))
        other = kpure(key, jnp.logical_not(b))
        est = ((-1) ** v) * (other - b_primal)
        return b_primal, b_tangent + est * p_tangent


flip_mvd = BernoulliMVD()


class BernoulliEnumParallel(ADEVPrimitive):
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
        ret_primals, ret_tangents = jax.vmap(kdual)(
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


flip_enum_parallel = BernoulliEnumParallel()


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
        ret_primals, ret_tangents = jax.vmap(kdual)((idxs,), (jnp.zeros_like(idxs),))

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


class NormalREPARAM(ADEVPrimitive):
    def sample(self, key, loc, scale_diag):
        return tfd.Normal(loc=loc, scale=scale_diag).sample(seed=key)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (mu_primal, sigma_primal) = primals
        (mu_tangent, sigma_tangent) = tangents
        eps = tfd.Normal(loc=0.0, scale=1.0).sample(seed=key)

        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return kdual((primal_out,), (tangent_out,))


normal_reparam = NormalREPARAM()


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

        eps = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=loc_primal.shape, seed=key
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

        return kdual((primal_out,), (tangent_out,))


mv_normal_diag_reparam = MvNormalDiagREPARAM()


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

        eps = tfd.Normal(loc=0.0, scale=1.0).sample(len(mu_primal), seed=key)

        def _inner(eps, mu, cov):
            L = jnp.linalg.cholesky(cov)
            return mu + L @ eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (eps, mu_primal, cov_primal),
            (jnp.zeros_like(eps), mu_tangent, cov_tangent),
        )
        return kdual((primal_out,), (tangent_out,))


mv_normal_reparam = MvNormalREPARAM()


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
        x = tfd.Uniform(low=0.0, high=1.0).sample(seed=key)
        return kdual((x,), (0.0,))


uniform = Uniform()


class Baseline(ADEVPrimitive):
    prim: ADEVPrimitive

    def sample(self, key, b, *args):
        return self.prim.sample(key, *args)

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (b_primal, *prim_primals) = primals
        (b_tangent, *prim_tangents) = tangents

        def new_kdual(v: Tuple, t: Tuple):
            ret_primal, ret_tangent = kdual(v, t)

            def _inner(ret, b):
                return ret - b

            return jax.jvp(
                _inner,
                (ret_primal, b_primal),
                (ret_tangent, b_tangent),
            )

        l_primal, l_tangent = self.prim.jvp_estimate(
            key,
            tuple(prim_primals),
            tuple(prim_tangents),
            (kpure, new_kdual),
        )

        def _inner(l, b):
            return l + b

        return jax.jvp(
            _inner,
            (l_primal, b_primal),
            (l_tangent, b_tangent),
        )


@typecheck
def baseline(prim):
    return Baseline(prim)


##################
# Loss primitive #
##################


class AddCost(ADEVPrimitive):
    def sample(self, *args):
        pass

    def jvp_estimate(
        self,
        key: PRNGKey,
        primals: Tuple,
        tangents: Tuple,
        konts: Tuple,
    ):
        (kpure, kdual) = konts
        (w,) = primals
        (w_tangent,) = tangents
        l_primal, l_tangent = kdual((), ())
        return l_primal + w, l_tangent + w_tangent


def add_cost(w):
    prim = AddCost()
    prim(w)

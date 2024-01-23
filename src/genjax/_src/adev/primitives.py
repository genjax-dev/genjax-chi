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
import jax.tree_util as jtu
from jax import util as jax_util
from jax.interpreters.ad import instantiate_zeros, recast_to_float0, zeros_like_jaxval
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    HigherOrderADEVPrimitive,
    batched_sample_p,
    sample,
    sample_p,
    sow_key,
)
from genjax._src.core.interpreters.forward import Environment
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
    Callable,
    FloatArray,
    Int,
    IntArray,
    List,
    PRNGKey,
    Tuple,
    dispatch,
    typecheck,
)

tfd = tfp.distributions


def zero(v):
    ad_zero = recast_to_float0(v, zeros_like_jaxval(v))
    return instantiate_zeros(ad_zero)


################################
# Gradient strategy primitives #
################################


# TODO: consider gradients as primitives.
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
        kpure, kdual = konts
        v = self.sample(key, *primals)
        l_primal, l_tangent = kdual((v,))
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
    def flatten(self):
        return (), ()

    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(
        self,
        _,
        primals,
        tangents,
        konts: Tuple[Callable, Callable],
    ):
        kpure, kdual = konts
        (p_primal,) = primals
        (p_tangent,) = tangents
        tl_primal, tl_tangent = kdual((True,))
        fl_primal, fl_tangent = kdual((False,))

        def _inner(p, tl, fl):
            return p * tl + (1 - p) * fl

        return jax.jvp(
            _inner,
            (p_primal, tl_primal, fl_primal),
            (p_tangent, tl_tangent, fl_tangent),
        )

    def get_batched_variant(self, batch_dims):
        return BatchedBernoulliEnumParallel(batch_dims)


flip_enum = BernoulliEnum()


class BernoulliMVD(ADEVPrimitive):
    def flatten(self):
        return (), ()

    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(
        self,
        key,
        primals,
        tangents,
        konts: Tuple[Callable, Callable],
    ):
        kpure, kdual = konts
        (p_primal,) = primals
        (p_tangent,) = tangents
        v = tfd.Bernoulli(probs=p_primal).sample(seed=key)
        b = v == 1
        b_primal, b_tangent = kdual((b,), (jnp.zeros_like(b),))
        other = kpure(jnp.logical_not(b))
        est = ((-1) ** v) * (other - b_primal)
        return b_primal, b_tangent + est * p_tangent

    def get_batched_variant(self, batch_dims):
        raise NotImplementedError


flip_mvd = BernoulliMVD()


class BernoulliEnumParallel(ADEVPrimitive):
    def sample(self, key, p):
        return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

    def jvp_estimate(self, _, primals, tangents, konts):
        kpure, kdual = konts
        (p_primal,) = primals
        (p_tangent,) = tangents
        ret_primals, ret_tangents = jax.vmap(kdual)((jnp.array([True, False]),))

        def _inner(p, ret):
            return jnp.sum(jnp.array([p, 1 - p]) * ret)

        return jax.jvp(
            _inner,
            (p_primal, ret_primals),
            (p_tangent, ret_tangents),
        )


flip_enum_parallel = BernoulliEnumParallel()


class BatchedBernoulliEnumParallel(ADEVPrimitive):
    batch_dims: Tuple = Pytree.static()

    def sample(self, key, p):
        def _inner(key, p):
            return 1 == tfd.Bernoulli(probs=p).sample(seed=key)

        return jax.vmap(_inner, in_axes=self.batch_dims)(key, p)

    def jvp_estimate(self, _, primals, tangents, konts):
        kpure, kdual = konts
        (p_primals,) = primals
        (p_tangents,) = tangents

        def iterated_outer_grid(array, N):
            grids = jnp.meshgrid(*([array] * N), indexing="ij")
            # Stack along a new axis and then reshape to 2D
            return jnp.stack(grids, axis=-1).reshape(-1, N)

        grid = iterated_outer_grid(jnp.array([True, False]), len(p_primals))
        ret_primals, ret_tangents = jax.vmap(kdual)((grid,))

        def _inner(ps, rets):
            def _sum(p, rets):
                return jnp.sum(jnp.array([p, 1 - p]) * rets)

            shaped_rets = rets.reshape(len(ps), -1)
            return jnp.sum(jax.vmap(_sum)(ps, shaped_rets))

        return jax.jvp(
            _inner,
            (p_primals, ret_primals),
            (p_tangents, zero(ret_tangents)),
        )


class CategoricalEnumParallel(ADEVPrimitive):
    def sample(self, key, probs):
        return tfd.Categorical(probs=probs).sample(seed=key)

    def jvp_estimate(self, _, primals, tangents, konts):
        kpure, kdual = konts
        (probs_primal,) = primals
        (probs_tangent,) = tangents
        idxs = jnp.arange(len(probs_primal))
        ret_primals, ret_tangents = jax.vmap(kdual)((idxs,))

        def _inner(probs, primals):
            return jnp.sum(jax.nn.softmax(probs) * primals)

        return jax.jvp(
            _inner,
            (probs_primal, ret_primals),
            (probs_tangent, ret_tangents),
        )

    @dispatch
    def __call__(self, probs: List):
        probs = jnp.log(jnp.array(probs))
        return sample(self, probs)

    @dispatch
    def __call__(self, probs: FloatArray):
        return sample(self, probs)


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

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
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

    def get_batched_variant(self, batch_dims):
        return BatchedNormalREPARAM(batch_dims)


normal_reparam = NormalREPARAM()


class BatchedNormalREPARAM(ADEVPrimitive):
    batch_dims: Tuple = Pytree.static()

    def sample(self, key, loc, scale_diag):
        return loc  # same type as the sample from MvNormalDiag.

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        (mu_primal, sigma_primal) = primals
        (mu_tangent, sigma_tangent) = tangents
        eps = tfd.Normal(
            loc=jnp.zeros_like(mu_primal),
            scale=jnp.ones_like(sigma_primal),
        ).sample(seed=key)

        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return kdual((primal_out,), (tangent_out,))


class MvNormalDiagREPARAM(ADEVPrimitive):
    def sample(self, key, loc, scale_diag):
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag).sample(
            seed=key
        )

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
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

    def get_batched_variant(self, batch_dims):
        return BatchedMvNormalDiagREPARAM(batch_dims)


mv_normal_diag_reparam = MvNormalDiagREPARAM()


class BatchedMvNormalDiagREPARAM(ADEVPrimitive):
    batch_dims: Tuple = Pytree.static()

    def sample(self, key, loc, scale_diag):
        return loc  # same type as the sample from MvNormalDiag.

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        (loc_primal, diag_scale_primal) = primals
        (loc_tangent, diag_scale_tangent) = tangents

        # eps.shape == loc_primal.shape
        eps = tfd.Normal(loc=0.0, scale=1.0).sample(
            sample_shape=len(loc_primal), seed=key
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


class MvNormalREPARAM(ADEVPrimitive):
    def sample(self, key, mu, sigma):
        v = tfd.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=sigma
        ).sample(seed=key)
        return v

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
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
    def sample(self, key):
        v = tfd.Uniform(low=0.0, high=1.0).sample(seed=key)
        return v

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        x = tfd.Uniform(low=0.0, high=1.0).sample(seed=key)
        return kdual((x,), (0.0,))

    def get_batched_variant(self, batch_dims):
        return BatchedUniform(batch_dims)


uniform = Uniform()


class BatchedUniform(ADEVPrimitive):
    def sample(self, key):
        return tfd.Uniform(low=0.0, high=1.0).sample(seed=key)

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        x = tfd.Uniform(low=0.0, high=1.0).sample(seed=key)
        return kdual((x,), (0.0,))


###########################
# Encapsulated primitives #
###########################


class Minibatch(ADEVPrimitive):
    f: Callable = Pytree.static()
    m: Int = Pytree.static()
    M: Int = Pytree.static()

    # TODO: check correctness.
    def jvp_estimate(self, key, primals, tangents, kont):
        M_range = jnp.arange(self.M)
        f_jvp = lambda primals, tangents: jax.jvp(self.f, primals, tangents)
        if self.M == 0:
            return kont.dual(0.0, 0.0)
        elif self.m == 0:
            primal = jnp.sum(jax.vmap(self.f)(M_range))
            tangent = jnp.sum(jax.vmap(f_jvp)(M_range, jnp.ones(self.M)))
            return kont.dual(primal, tangent)
        else:
            selected_idxs = tfd.FiniteDiscrete(M_range, probs=jnp.ones(self.M)).sample(
                self.M, seed=key
            )
            primal = (self.M / self.m) * jnp.sum(jax.vmap(self.f)(selected_idxs))
            tangent = (self.M / self.m) * jnp.sum(
                jax.vmap(f_jvp)(selected_idxs, jnp.ones(self.m))
            )
            return kont.dual(primal, tangent)


@typecheck
def minibatch(f: Callable, m: Int, M: Int):
    return Minibatch.new(f, m, M)


class Average(ADEVPrimitive):
    p: ADEVPrimitive
    N: IntArray = Pytree.static()

    def jvp_estimate(self, key, primals, tangents, kont):
        sub_keys = jax.random.split(key, self.N)
        v, tangent = jax.vmap(
            self.p.jvp_estimate,
            in_axes=(0, None, None, None),
        )(sub_keys, primals, tangents, kont)
        return jnp.mean(v), jnp.mean(tangent)


@typecheck
def average(p: ADEVPrimitive, N: IntArray):
    return Average(N, p)


class Baseline(ADEVPrimitive):
    prim: ADEVPrimitive

    def sample(self, key, b, *args):
        return self.prim.sample(key, *args)

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        (b_primal, *prim_primals) = primals
        (b_tangent, *prim_tangents) = tangents

        @dispatch
        def new_kdual(v: Tuple):
            ret_primal, ret_tangent = kdual(v)

            def _inner(ret, b):
                return ret - b

            dual = jax.jvp(
                _inner,
                (ret_primal, b_primal),
                (ret_tangent, b_tangent),
            )
            return dual

        @dispatch
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


###########################
# Higher order primitives #
###########################


class Maps(HigherOrderADEVPrimitive):
    callable: Callable = Pytree.static()

    def _transform_jaxpr(self, jaxpr, consts, flat_args):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.invars, flat_args)
        jax_util.safe_map(env.write, jaxpr.constvars, consts)

        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            if eqn.primitive == sample_p:
                # We swap `sample_p` with `batched_sample_p`.
                outvals = batched_sample_p.bind(*args, **params)
            else:
                outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)
        return jax_util.safe_map(env.read, jaxpr.outvars)

    def _transform(self, *args):
        f = self.callable
        closed_jaxpr, (flat_args, _, out_tree) = stage(f)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        out = self._transform_jaxpr(jaxpr, consts, flat_args)
        return jtu.tree_unflatten(out_tree(), out)

    def transform(self, key, *args):
        def wrapped(key, args):
            return sow_key(self._transform)(key, *args)

        return jax.vmap(wrapped, in_axes=(None, 0))(key, args)


@typecheck
def maps(callable: Callable):
    return Maps.new(callable)


##################
# Loss primitive #
##################


class AddCost(ADEVPrimitive):
    def sample(self, key, *args):
        pass

    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts
        (w,) = primals
        (w_tangent,) = tangents
        l_primal, l_tangent = kdual(())
        return l_primal + w, l_tangent + w_tangent


def add_cost(w):
    prim = AddCost()
    prim(w)

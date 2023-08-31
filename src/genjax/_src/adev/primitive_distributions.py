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
"""Defines and registers ADEV primitives for several `Distribution` generative
functions."""

import dataclasses

import jax
import jax.numpy as jnp

from genjax._src.adev.lang import ADEVPrimitive
from genjax._src.adev.lang import SupportsEnum
from genjax._src.adev.lang import SupportsMVD
from genjax._src.adev.lang import SupportsREINFORCE
from genjax._src.adev.lang import register
from genjax._src.generative_functions.distributions.scipy.bernoulli import Bernoulli
from genjax._src.generative_functions.distributions.scipy.bernoulli import bernoulli
from genjax._src.generative_functions.distributions.scipy.normal import Normal
from genjax._src.generative_functions.distributions.scipy.normal import normal
from genjax._src.generative_functions.distributions.scipy.poisson import Poisson
from genjax._src.generative_functions.distributions.scipy.poisson import poisson


identity = lambda v: v

#####
# Normal
#####


@dataclasses.dataclass
class ADEVPrimNormal(ADEVPrimitive, SupportsMVD, SupportsREINFORCE):
    def sample(self, key, args):
        tr = normal.simulate(key, args)
        v = tr.get_retval()
        return v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        (μ, σ) = primals
        (μ_tangent, σ_tangent) = tangents
        key, sub_key = jax.random.split(key)
        v = normal.sample(sub_key, μ, σ)
        lp = normal.logpdf(v, μ, σ)
        r_primal, r_tangent = kont([v], [1.0])
        return [r_primal], [r_tangent * lp]

    def mvd_estimate(self, key, duals, kont):
        raise NotImplementedError


register(Normal, ADEVPrimNormal)

#####
# Bernoulli
#####


@dataclasses.dataclass
class ADEVPrimBernoulli(
    ADEVPrimitive,
    SupportsEnum,
    SupportsREINFORCE,
):
    def sample(self, key, args):
        (p,) = args
        v = bernoulli.sample(key, p)
        return v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def enum_exact(self, _, primals, tangents, kont):
        (p,) = primals
        (p_tangent,) = tangents
        tl_primal, tl_tangent = kont([True], [jnp.array(0.0)])
        fl_primal, fl_tangent = kont([False], [jnp.array(0.0)])
        ev_primal = p * tl_primal + (1 - p) * fl_primal
        return [ev_primal], [
            (p_tangent * tl_primal)
            + (p * tl_tangent)
            - (p_tangent * fl_primal)
            + (1 - p) * fl_tangent
        ]

    def reinforce_estimate(self, key, primals, tangents, kont):
        (p,) = primals
        (p_tangent,) = tangents
        b = bernoulli.sample(key, p)
        l_primal, l_tangent = kont([b], [jnp.array(0.0)])
        lp_tangent = jax.lax.cond(
            b,
            lambda *_: p_tangent * (1 / p),
            lambda *_: -p_tangent * (1 / (1 - p)),
        )
        return [l_primal], [l_tangent + l_primal * lp_tangent]


register(Bernoulli, ADEVPrimBernoulli)

#####
# Poisson
#####


@dataclasses.dataclass
class ADEVPrimPoisson(ADEVPrimitive, SupportsMVD, SupportsREINFORCE):
    def sample(self, key, args):
        key, tr = poisson.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(
            self,
            key,
            primals,
            tangents,
            identity,
        )

    def reinforce_estimate(self, key, primals, tangents, kont):
        pass

    def mvd_estimate(self, key, primals, _, kont):
        (θ,) = primals
        x_minus = poisson.sample(key, θ)
        x_plus = x_minus + 1
        y_plus = kont(x_plus)
        y_minus = kont(x_minus)
        nabla_est = y_minus - y_plus
        return nabla_est


register(Poisson, ADEVPrimPoisson)

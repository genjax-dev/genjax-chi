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

from genjax._src.core.transforms.adev import ADEVPrimitive
from genjax._src.core.transforms.adev import SupportsMVD
from genjax._src.core.transforms.adev import SupportsREINFORCE
from genjax._src.core.transforms.adev import register
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
    def simulate(self, key, args):
        key, tr = normal.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        (μ, σ) = primals
        (μ_tangent, σ_tangent) = tangents
        key, sub_key = jax.random.split(key)
        v = normal.sample(sub_key, μ, σ)
        lp = normal.logpdf(v, μ, σ)
        key, r_primal, r_tangent = kont(key, [v], [1.0])
        return key, [r_primal], [r_tangent * lp]

    def mvd_estimate(self, key, duals, kont):
        raise NotImplementedError


register(Normal, ADEVPrimNormal)

#####
# Bernoulli
#####


@dataclasses.dataclass
class ADEVPrimBernoulli(ADEVPrimitive, SupportsREINFORCE):
    def simulate(self, key, args):
        key, tr = bernoulli.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        (p,) = primals
        (p_tangent,) = tangents
        key, sub_key = jax.random.split(key)
        key, b = bernoulli.sample(sub_key, p)
        retdual = kont(b)
        l_primal, l_tangent = retdual.primal, retdual.tangent
        dlp = jax.lax.switch(
            b,
            lambda *_: jnp.log(p_tangent),
            lambda *_: jnp.log(1 - p_tangent),
        )
        return key, (l_primal, l_tangent + l_primal * dlp.tangent)


register(Bernoulli, ADEVPrimBernoulli)

#####
# Poisson
#####


@dataclasses.dataclass
class ADEVPrimPoisson(ADEVPrimitive, SupportsMVD, SupportsREINFORCE):
    def simulate(self, key, args):
        key, tr = poisson.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        pass

    def mvd_estimate(self, key, primals, _, kont):
        (θ,) = primals
        key, sub_key = jax.random.split(key)
        key, x_minus = poisson.sample(sub_key, θ)
        x_plus = x_minus + 1
        y_plus = kont(x_plus)
        y_minus = kont(x_minus)
        nabla_est = y_minus - y_plus
        return nabla_est


register(Poisson, ADEVPrimPoisson)

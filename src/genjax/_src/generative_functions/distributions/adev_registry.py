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
from genjax._src.core.transforms.adev import SupportsEnum
from genjax._src.core.transforms.adev import SupportsMVD
from genjax._src.core.transforms.adev import SupportsReinforce
from genjax._src.core.transforms.adev import register
from genjax._src.generative_functions.distributions.scipy.bernoulli import Bernoulli
from genjax._src.generative_functions.distributions.scipy.bernoulli import _Bernoulli
from genjax._src.generative_functions.distributions.scipy.normal import Normal
from genjax._src.generative_functions.distributions.scipy.normal import _Normal
from genjax._src.generative_functions.distributions.scipy.poisson import Poisson
from genjax._src.generative_functions.distributions.scipy.poisson import _Poisson


identity = lambda v: v

#####
# Normal
#####


@dataclasses.dataclass
class ADEVPrimNormal(ADEVPrimitive, SupportsMVD, SupportsReinforce):
    def simulate(self, key, args):
        key, tr = Normal.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        pass

    def mvd_estimate(self, key, duals, kont):
        pass


register(_Normal, ADEVPrimNormal)

#####
# Bernoulli
#####


@dataclasses.dataclass
class ADEVPrimBernoulli(ADEVPrimitive, SupportsEnum, SupportsReinforce):
    def simulate(self, key, args):
        key, tr = Normal.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        (p,) = primals
        (p_tangent,) = tangents
        key, sub_key = jax.random.split(key)
        key, b = Bernoulli.sample(sub_key, p)
        retdual = kont(b)
        l_primal, l_tangent = retdual.primal, retdual.tangent
        dlp = jax.lax.switch(
            b,
            lambda *_: jnp.log(p_tangent),
            lambda *_: jnp.log(1 - p_tangent),
        )
        return l_primal, l_tangent + l_primal * dlp.tangent


register(_Bernoulli, ADEVPrimBernoulli)

#####
# Poisson
#####


@dataclasses.dataclass
class ADEVPrimPoisson(ADEVPrimitive, SupportsMVD, SupportsReinforce):
    def simulate(self, key, args):
        key, tr = Poisson.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def grad_estimate(self, key, primals, tangents):
        return self.reinforce_estimate(self, key, primals, tangents, identity)

    def reinforce_estimate(self, key, primals, tangents, kont):
        pass

    def mvd_estimate(self, key, primals, _, kont):
        (θ,) = primals
        key, sub_key = jax.random.split(key)
        key, x_minus = Poisson.sample(sub_key, θ)
        x_plus = x_minus + 1
        y_plus = kont(x_plus)
        y_minus = kont(x_minus)
        nabla_est = y_minus - y_plus
        return nabla_est


register(_Poisson, ADEVPrimPoisson)

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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from adevjax import ADEVPrimitive
from adevjax import E
from adevjax import add_cost
from adevjax import adev
from adevjax import flip_enum
from adevjax import geometric_reinforce
from adevjax import grab_key
from adevjax import mv_normal_diag_reparam
from adevjax import mv_normal_reparam
from adevjax import normal_reinforce
from adevjax import normal_reparam
from adevjax import sample_with_key

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Callable
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_bernoulli,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_mv_normal,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_mv_normal_diag,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_normal,
)
from genjax._src.gensp.target import Target


##########################################
# Differentiable distribution primitives #
##########################################


@dataclass
class ADEVDistribution(ExactDensity):
    differentiable_logpdf: Callable
    adev_primitive: ADEVPrimitive

    def flatten(self):
        return (self.adev_primitive,), (self.differentiable_logpdf,)

    @classmethod
    def new(cls, adev_prim, diff_logpdf):
        return ADEVDistribution(diff_logpdf, adev_prim)

    def sample(self, key, *args):
        return sample_with_key(self.adev_primitive, key, *args)

    def logpdf(self, v, *args):
        return self.differentiable_logpdf(v, *args)


flip_enum = ADEVDistribution.new(
    flip_enum,
    lambda v, p: tfp_bernoulli.logpdf(v, probs=p),
)

normal_reinforce = ADEVDistribution.new(
    normal_reinforce,
    lambda v, μ, σ: tfp_normal.logpdf(v, μ, σ),
)

normal_reparam = ADEVDistribution.new(
    normal_reparam,
    lambda v, μ, σ: tfp_normal.logpdf(v, μ, σ),
)

mv_normal_reparam = ADEVDistribution.new(
    mv_normal_reparam,
    lambda v, μ, Σ: tfp_mv_normal.logpdf(v, μ, Σ),
)

mv_normal_diag_reparam = ADEVDistribution.new(
    mv_normal_diag_reparam,
    lambda v, μ, Σ_diag: tfp_mv_normal_diag.logpdf(v, μ, Σ_diag),
)

geometric_reinforce = ADEVDistribution.new(
    geometric_reinforce,
    lambda v, *args: tfp_geometric.logpdf(v, *args),
)

##################################
# Differentiable loss primitives #
##################################


def upper(prim: Distribution):
    def _inner(key, *args):
        (w, v) = prim.random_weighted(key, *args)
        add_cost(-w)
        return v

    return lambda key, *args: _inner(key, *args)


def lower(prim: Distribution):
    def _inner(key, v, *args):
        w = prim.estimate_logpdf(key, v, *args)
        add_cost(w)

    return lambda key, v, *args: _inner(key, v, *args)


def loss(fn: Callable):
    @adev
    def _inner(*args):
        v = fn(*args)
        if v is None:
            return 0.0
        else:
            return v

    return E(_inner)

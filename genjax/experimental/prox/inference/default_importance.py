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

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from genjax.experimental.prox.utilities import logsumexp_with_extra


@dataclass
class DefaultImportance(ProxDistribution):
    num_particles: int

    def flatten(self):
        return (), (self.num_particles,)

    @classmethod
    def unflatten(cls, xs, data):
        return DefaultImportance(*xs, *data)

    def random_weighted(self, key, target: Target):
        key, sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (lws, tr) = jax.vmap(target.p.importance, in_axes=(0, None, None))(
            sub_keys, target.constraints, target.args
        )
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        key, sub_key = jax.random.split(key)
        index = jax.random.categorical(sub_key, lnw)
        selected_particle = tr.slice(index)
        selected_lw = lws[index]
        return key, (target.get_latents(tr), selected_particle.get_score() - aw)

    def estimate_logpdf(self, key, chm, target):
        key, sub_keys = jax.random.split(key, self.num_particles)
        sub_keys = jnp.array(sub_keys)
        _, (lws, tr) = jax.vmap(target.p.importance, in_axes=(0, None, None))(
            sub_keys, target.constraints, target.args
        )
        merged = chm.merge(target.constraints)
        key, retained_tr = target.p.importance(key, merged, target.args)
        constrained = target.constraints.to_selection()
        _, retained_w = constrained.filter(retained_tr)
        lse = logsumexp_with_extra(lws, retained_w)
        return retained_tr.get_score() - lse + np.log(self.num_particles)

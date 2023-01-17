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

import dataclasses
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp

from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck
from genjax._src.utilities import slash


@dataclasses.dataclass
class ImportanceSampling(Pytree):
    num_particles: IntArray
    model: GenerativeFunction
    proposal: Union[None, GenerativeFunction] = None

    def flatten(self):
        return (), (self.num_particles, self.model, self.proposal)

    @typecheck
    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
        proposal: Union[None, GenerativeFunction] = None,
    ):
        return ImportanceSampling(
            num_particles,
            model,
            proposal=proposal,
        )

    def _bootstrap_importance_sampling(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
    ):
        key, sub_keys = slash(key, self.num_particles + 1)
        _, (lws, trs) = jax.vmap(
            self.model.importance, in_axes=(0, None, None)
        )(
            sub_keys,
            observations,
            model_args,
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        return key, (trs, log_normalized_weights, log_ml_estimate)

    def _proposal_importance_sampling(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        proposal_args: Tuple,
    ):
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, p_trs = jax.vmap(self.proposal.simulate, in_axes=(0, None, None))(
            sub_keys,
            observations,
            proposal_args,
        )
        observations = jax.tree_util.map(
            lambda v: jnp.repeats(v, self.num_particles), observations
        )
        chm = p_trs.get_choices().merge(observations)
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (lws, m_trs) = jax.vmap(
            self.model.importance, in_axes=(0, 0, None)
        )(
            sub_keys,
            chm,
            model_args,
        )
        lws = lws - p_trs.get_score()
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        return key, (m_trs, log_normalized_weights, log_ml_estimate)

    @typecheck
    def apply(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        # Importance sampling with custom proposal branch.
        if len(args) == 2:
            assert isinstance(args[0], tuple)
            assert isinstance(args[1], tuple)
            assert self.proposal is not None
            model_args = args[0]
            proposal_args = args[1]
            return self._proposal_importance_sampling(
                key, choice_map, model_args, proposal_args
            )
        # Bootstrap importance sampling branch.
        else:
            assert isinstance(args, tuple)
            assert self.proposal is None
            model_args = args[0]
            return self._bootstrap_importance_sampling(
                key, choice_map, model_args
            )

    @typecheck
    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


@dataclasses.dataclass
class SamplingImportanceResampling(Pytree):
    num_particles: IntArray
    model: GenerativeFunction
    proposal: Union[None, GenerativeFunction] = None

    def flatten(self):
        return (), (self.num_particles, self.model, self.proposal)

    @classmethod
    def new(
        cls,
        num_particles: IntArray,
        model: GenerativeFunction,
        proposal: Union[None, GenerativeFunction] = None,
    ):
        return SamplingImportanceResampling(
            num_particles,
            model,
            proposal=proposal,
        )

    def _bootstrap_importance_resampling(
        self,
        key: PRNGKey,
        obs: ChoiceMap,
        model_args: Tuple,
    ):
        key, sub_keys = slash(key, self.num_particles)
        _, (lws, trs) = jax.vmap(
            self.model.importance, in_axes=(0, None, None)
        )(sub_keys, obs, model_args)
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(self.num_particles)
        key, sub_key = jax.random.split(key)
        ind = jax.random.categorical(sub_key, log_normalized_weights)
        tr = jax.tree_util.tree_map(lambda v: v[ind], trs)
        lnw = log_normalized_weights[ind]
        return key, (tr, lnw, log_ml_estimate)

    def apply(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        # Importance resampling with custom proposal branch.
        if len(args) == 2:
            assert isinstance(args[0], tuple)
            assert isinstance(args[1], tuple)
            assert self.proposal is not None
            model_args = args[0]
            proposal_args = args[1]
            return self._proposal_importance_resampling(
                key, choice_map, model_args, proposal_args
            )
        # Bootstrap importance resampling branch.
        else:
            assert isinstance(args, tuple)
            assert self.proposal is None
            model_args = args[0]
            return self._bootstrap_importance_resampling(
                key, choice_map, model_args
            )

    def __call__(self, key: PRNGKey, choice_map: ChoiceMap, *args):
        return self.apply(key, choice_map, *args)


##############
# Shorthands #
##############

importance_sampling = ImportanceSampling.new
sampling_importance_resampling = SamplingImportanceResampling.new
sir = sampling_importance_resampling

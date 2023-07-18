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
"""A module containing a test suite for inference based on exact inference in
hidden Markov models (HMMs)."""

from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.generative_functions.combinators.vector.unfold_combinator import Unfold
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    vector_select,
)
from genjax._src.generative_functions.distributions.custom.discrete_hmm import (
    DiscreteHMM,
)
from genjax._src.generative_functions.distributions.custom.discrete_hmm import (
    DiscreteHMMConfiguration,
)
from genjax._src.generative_functions.distributions.custom.discrete_hmm import (
    discrete_hmm_config,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_categorical,
)
from genjax._src.language_decorator import gen


def build_inference_test_generator(
    max_length: IntArray,
    state_space_size: IntArray,
    transition_distance_truncation: IntArray,
    observation_distance_truncation: IntArray,
    transition_variance: FloatArray,
    observation_variance: FloatArray,
):
    config = discrete_hmm_config(
        state_space_size,
        transition_distance_truncation,
        observation_distance_truncation,
        transition_variance,
        observation_variance,
    )

    @gen(Unfold, max_length=max_length)
    def markov_chain(state: IntArray, config: DiscreteHMMConfiguration):
        transition = config.transition_tensor
        observation = config.observation_tensor
        z = tfp_categorical(transition[state, :]) @ "z"
        _ = tfp_categorical(observation[z, :]) @ "x"
        return z

    def inference_test_generator(key: PRNGKey, initial_state: IntArray):
        key, tr = markov_chain.simulate(key, (max_length - 1, initial_state, config))
        z_sel = vector_select("z")
        x_sel = vector_select("x")
        latent_sequence = z_sel.filter(tr)["z"]
        observation_sequence = x_sel.filter(tr)["x"]
        log_data_marginal = DiscreteHMM.data_logpdf(config, observation_sequence)
        # This actually doesn't use any randomness.
        key, (log_posterior, _) = DiscreteHMM.estimate_logpdf(
            key, latent_sequence, config, observation_sequence
        )
        return key, (
            log_posterior,
            log_data_marginal,
            latent_sequence,
            observation_sequence,
        )

    return inference_test_generator

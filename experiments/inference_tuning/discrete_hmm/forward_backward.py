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
import genjax
import genjax.experimental.prox as prox
from tensorflow_probability.substrates import jax as tfp
from model_config import DiscreteHMMConfiguration, hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
)
from dataclasses import dataclass

tfd = tfp.distributions


def latent_marginals(config: DiscreteHMMConfiguration, observations):
    init = int(config.linear_grid_dim / 2)
    initial_distribution = tfd.Categorical(
        logits=config.transition_tensor[init, :]
    )
    transition_distribution = tfd.Categorical(logits=config.transition_tensor)
    observation_distribution = tfd.Categorical(logits=config.observation_tensor)
    hmm = tfd.HiddenMarkovModel(
        initial_distribution,
        transition_distribution,
        observation_distribution,
        len(observations),
    )
    marginals = hmm.posterior_marginals(observations)
    return hmm, marginals


def log_data_marginal(config, observations):
    hmm, _ = latent_marginals(config, observations)
    return hmm.log_prob(observations)


def latent_sequence_sample(key, config: DiscreteHMMConfiguration, observations):
    _, marginals = latent_marginals(config, observations)

    def _inner(carry, index):
        key, prev = carry
        probs = jnp.matmul(jax.nn.softmax(config.transition_tensor), prev)
        key, sub_key = jax.random.split(key)
        v = jax.random.categorical(sub_key, jnp.log(probs))
        return (key, probs), v

    (key, _), v = jax.lax.scan(
        _inner,
        (key, jax.nn.softmax(marginals[0].logits)),
        None,
        length=len(observations),
    )
    return key, v


def latent_sequence_posterior(
    config: DiscreteHMMConfiguration, latent_point, observations
):
    hmm, marginals = latent_marginals(config, observations)

    def _inner(carry, x):
        latent, obs = x
        v = jnp.log(carry[latent])
        v += jnp.log(
            jax.nn.softmax(hmm.observation_distribution.logits)[latent, obs]
        )
        carry = jax.nn.softmax(hmm.transition_distribution.logits[latent, :])
        return carry, v

    _, probs = jax.lax.scan(
        _inner,
        jax.nn.softmax(hmm.initial_distribution.logits),
        (latent_point, observations),
    )
    prod = jnp.sum(probs)
    prod -= hmm.log_prob(observations)
    return prod, (probs, hmm.log_prob(observations))


@dataclass
class _ExactHMMPosterior(genjax.Distribution):
    def flatten(self):
        return (), ()

    def sample(self, key, config, observation_sequence):
        _, v = latent_sequence_sample(key, config, observation_sequence)
        return v

    def logpdf(self, key, v, config, observation_sequence):
        prob, _ = latent_sequence_posterior(config, v, observation_sequence)
        return jnp.log(prob)

    def data_logpdf(self, config, observation_sequence):
        return log_data_marginal(config, observation_sequence)


ExactHMMPosterior = _ExactHMMPosterior()

#####
# Test custom_smc
#####


key = jax.random.PRNGKey(314159)
num_steps = 2
config = DiscreteHMMConfiguration.new(2, 1, 1, 0.01, 0.01)
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
    key, (num_steps, config)
)

# Observations.
observation_sequence = tr["z", "observation"]
chm_sequence = genjax.VectorChoiceMap.new(
    np.array([ind for ind in range(0, len(observation_sequence))]),
    genjax.ChoiceMap.new(
        {("z", "observation"): observation_sequence},
    ),
)

num_steps = len(observation_sequence)

# Inference target.
final_target = prox.Target(
    hidden_markov_model,
    None,
    (num_steps, config),
    chm_sequence,
)

# Latent choice map.
chm = genjax.ValueChoiceMap.new(
    genjax.VectorChoiceMap.new(
        np.array([ind for ind in range(0, len(observation_sequence))]),
        genjax.ChoiceMap.new({("z", "latent"): observation_sequence}),
    )
)

custom_smc = genjax.CustomSMC(
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
    lambda _: num_steps,
    2000,
)

n_samples = 100
key, *sub_keys = jax.random.split(key, n_samples + 1)
sub_keys = jnp.array(sub_keys)
key, (w, tr) = jax.jit(
    jax.vmap(custom_smc.importance, in_axes=(0, None, None))
)(sub_keys, chm, (final_target,))
w = jax.scipy.special.logsumexp(w) - jnp.log(n_samples)

probs, seq = latent_sequence_posterior(
    config,
    observation_sequence,
    observation_sequence,
)
print(w)
print(probs)
print(seq)

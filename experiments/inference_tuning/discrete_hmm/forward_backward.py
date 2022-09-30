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

genjax.go_pretty()

tfd = tfp.distributions

#####
# Forward-filtering backward sampling
#####

# This implements JAX compatible forward-filtering
# backward sampling (to produce exact samples from discrete HMM
# posteriors)


def forward_filtering_backward_sampling(
    key, config: DiscreteHMMConfiguration, observation_sequence
):
    init = int(config.linear_grid_dim / 2)
    prior = jnp.log(jax.nn.softmax(config.transition_tensor[init, :]))
    transition_n = jnp.log(jax.nn.softmax(config.transition_tensor))
    obs_n = jnp.log(jax.nn.softmax(config.observation_tensor))

    # Computing the alphas and forward filter distributions:
    #
    # \alpha_1(x_1) = p(x_1) * p(y_1 | x_1) [[ initialization ]]
    #
    # \alpha_t(x_t) = p(y_t | x_t) * \sum_{x_{t-1}=1}^N [ p(x_{t-1}, y_1, ..., y_{t-1})
    #                              * p(x_t | x_{t-1}) ]
    #               = p(y_t | x_t) * \sum_{x_{t-1}=1}^N \alpha_t(x_t) * p(x_t | x_{t-1})
    #                                for t=2, .., T

    def forward_pass(carry, x):
        index, prev = carry
        obs = x

        def t_branch(prev, obs):
            alpha = jax.scipy.special.logsumexp(
                prev + transition_n,
                axis=-1,
            )
            alpha = obs_n + alpha
            alpha = alpha[obs, :]
            return alpha

        def init_branch(prev, obs):
            broadcasted = obs_n + prev
            alpha = jax.scipy.special.logsumexp(obs_n + prev, axis=1)
            assert False
            alpha = alpha[obs, :]
            return alpha

        check = index == 0
        alpha = jax.lax.cond(check, init_branch, t_branch, prev, obs)
        forward_filter = alpha - jax.scipy.special.logsumexp(alpha)
        return (index + 1, alpha), (alpha, forward_filter)

    _, (alpha, forward_filters) = jax.lax.scan(
        forward_pass, (0, prior), observation_sequence
    )

    # Computing the backward distributions.
    # Start by sampling from x_T from p(x_T | y_1, .., y_T)
    # which is the last forward filter distribution.
    #
    # Then:
    # p(x_{t-1} | x_t, y_{1:T}) = p(x_{t-1} | y_{1:t-1}) * p(x_t | x_{t-1}) / Z_t
    # where Z_t = \sum_{x_{t-1}=1}^N p(x_{t-1} | y_{1:t-1}) * p(x_t | x_{t-1})

    def backward_sample(carry, x):
        key, index, prev_sample = carry
        forward_filter = x

        def end_branch(key, prev, forward_filter):
            sample = jax.random.categorical(key, forward_filter)
            return sample

        def t_1_branch(key, prev, forward_filter):
            backward_distribution = (
                forward_filter + transition_n[:, prev_sample]
            )
            backward_distribution = (
                backward_distribution
                - jax.scipy.special.logsumexp(backward_distribution)
            )
            sample = jax.random.categorical(key, backward_distribution)
            return sample

        key, sub_key = jax.random.split(key)
        check = index == 0
        sample = jax.lax.cond(
            check, end_branch, t_1_branch, sub_key, prev_sample, forward_filter
        )
        return (key, index + 1, sample), sample

    # This is supposed to be scanned in reverse order
    # from the forward order.
    (key, _, _), samples = jax.lax.scan(
        backward_sample,
        (key, 0, 0),
        jnp.flip(forward_filters, axis=0),
    )
    return key, samples


#####
# ExactHMMPosterior
#####


def latent_marginals(config: DiscreteHMMConfiguration, observation_sequence):
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
        len(observation_sequence),
    )
    marginals = hmm.posterior_marginals(observation_sequence)
    return hmm, marginals


def log_data_marginal(config, observation_sequence):
    hmm, _ = latent_marginals(config, observation_sequence)
    return hmm.log_prob(observation_sequence)


def latent_sequence_sample(
    key, config: DiscreteHMMConfiguration, observation_sequence
):
    _, marginals = latent_marginals(config, observation_sequence)

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
        length=len(observation_sequence),
    )
    return key, v


def latent_sequence_posterior(
    config: DiscreteHMMConfiguration, latent_point, observation_sequence
):
    hmm, marginals = latent_marginals(config, observation_sequence)

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
        (latent_point, observation_sequence),
    )
    prod = jnp.sum(probs)
    prod -= hmm.log_prob(observation_sequence)
    return prod, (probs, hmm.log_prob(observation_sequence))


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
num_steps = 20
config = DiscreteHMMConfiguration.new(20, 1, 1, 0.2, 0.05)
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
    key, (num_steps, config)
)
observation_sequence = tr["z", "observation"]
key, sample = jax.jit(forward_filtering_backward_sampling)(
    key, config, observation_sequence
)
print(sample)


def test_exact_marginal(benchmark):
    key = jax.random.PRNGKey(314159)
    num_steps = 20
    config = DiscreteHMMConfiguration.new(20, 1, 1, 0.2, 0.05)
    key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
        key, (num_steps, config)
    )

    # observation_sequence.
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
        50,
    )

    n_samples = 100
    key, *sub_keys = jax.random.split(key, n_samples + 1)
    sub_keys = jnp.array(sub_keys)
    key, (marg, tr) = jax.jit(
        jax.vmap(custom_smc.importance, in_axes=(0, None, None))
    )(sub_keys, chm, (final_target,))
    marg = jax.scipy.special.logsumexp(marg) - jnp.log(n_samples)

    score = log_data_marginal(
        config,
        observation_sequence,
    )

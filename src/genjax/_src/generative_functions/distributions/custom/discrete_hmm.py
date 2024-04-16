# Copyright 2024 MIT Probabilistic Computing Project
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
from scipy.linalg import circulant
from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import FloatArray, IntArray, PRNGKey
from genjax._src.generative_functions.distributions.distribution import Distribution

tfd = tfp.distributions

#####
# Discrete HMM configuration
#####


def scaled_circulant(N, k, epsilon, delta):
    source = [
        epsilon ** abs(index)
        if index <= k
        else epsilon ** abs(index - N)
        if index - N >= -k
        else -delta
        for index in range(0, N)
    ]
    return circulant(source)


class DiscreteHMMConfiguration(Pytree):
    linear_grid_dim: IntArray = Pytree.static()
    adjacency_distance_trans: IntArray = Pytree.static()
    adjacency_distance_obs: IntArray = Pytree.static()
    sigma_trans: FloatArray = Pytree.static()
    sigma_obs: FloatArray = Pytree.static()

    def flatten(self):
        return (), (
            self.linear_grid_dim,
            self.adjacency_distance_trans,
            self.adjacency_distance_obs,
            self.sigma_trans,
            self.sigma_obs,
        )

    @classmethod
    def copy(cls, config, transition_tensor, observation_tensor):
        return DiscreteHMMConfiguration(
            config.linear_grid_dim,
            config.adjacency_distance_trans,
            config.adjacency_distance_obs,
            config.sigma_trans,
            config.sigma_obs,
        )

    def transition_tensor(self):
        return scaled_circulant(
            self.linear_grid_dim,
            self.adjacency_distance_trans,
            self.sigma_trans if self.sigma_trans > 0.0 else -np.inf,
            1 / self.sigma_trans if self.sigma_trans > 0.0 else -np.inf,
        )

    def observation_tensor(self):
        return scaled_circulant(
            self.linear_grid_dim,
            self.adjacency_distance_obs,
            self.sigma_obs if self.sigma_obs > 0.0 else -np.inf,
            1 / self.sigma_obs if self.sigma_obs > 0.0 else np.inf,
        )


#####
# Forward-filtering backward sampling
#####

# This implements JAX compatible forward-filtering
# backward sampling (to produce exact samples from discrete HMM
# posteriors)


def forward_filtering_backward_sampling(
    key: PRNGKey, config: DiscreteHMMConfiguration, observation_sequence
):
    init = int(config.linear_grid_dim / 2)
    tt = config.transition_tensor()
    prior = jnp.log(jax.nn.softmax(tt[init, :]))
    transition_n = jnp.log(jax.nn.softmax(tt))
    obs_n = jnp.log(jax.nn.softmax(config.observation_tensor()))

    # Computing the alphas and forward filter distributions:
    #
    # \alpha_1(x_1) = p(x_1) * p(y_1 | x_1) [[ initialization ]]
    #
    # \alpha_t(x_t) = p(y_t | x_t) * \sum_{x_{t-1}=1}^N
    #           [ p(x_{t-1}, y_1, ..., y_{t-1})* p(x_t | x_{t-1}) ]
    #
    #               = p(y_t | x_t)
    #           * \sum_{x_{t-1}=1}^N \alpha_t(x_t) * p(x_t | x_{t-1})
    #                                for t=2, .., T

    def forward_pass(carry, x):
        index, prev = carry
        obs = x

        def t_branch(prev, obs):
            alpha = jax.scipy.special.logsumexp(
                prev + transition_n,
                axis=-1,
            )
            alpha = obs_n + alpha.reshape(-1, 1)
            alpha = alpha[:, obs]
            return alpha

        def init_branch(prev, obs):
            alpha = obs_n + prev.reshape(-1, 1)
            alpha = alpha[:, obs]
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
    # p(x_{t-1} | x_t, y_{1:T}) = p(x_{t-1} | y_{1:t-1})
    #                                   * p(x_t | x_{t-1}) / Z_t
    #
    # where Z_t = \sum_{x_{t-1}=1}^N p(x_{t-1} | y_{1:t-1})
    #                                   * p(x_t | x_{t-1})

    def backward_sample(carry, x):
        key, index, prev_sample = carry
        forward_filter = x

        def end_branch(key, prev, forward_filter):
            sample = jax.random.categorical(key, forward_filter)
            return sample

        def t_1_branch(key, prev, forward_filter):
            backward_distribution = forward_filter + transition_n[:, prev_sample]
            backward_distribution = backward_distribution - jax.scipy.special.logsumexp(
                backward_distribution
            )
            sample = jax.random.categorical(key, backward_distribution)
            return sample

        key, sub_key = jax.random.split(key)
        check = index == 0
        sample = jax.lax.cond(
            check,
            end_branch,
            t_1_branch,
            sub_key,
            prev_sample,
            forward_filter,
        )
        return (key, index + 1, sample), sample

    # This is supposed to be scanned in reverse order
    # from the forward order.
    (key, _, _), samples = jax.lax.scan(
        backward_sample,
        (key, 0, 0),
        jnp.flip(forward_filters, axis=0),
    )
    samples = jnp.flip(samples)
    return key, (samples, forward_filters)


#####
# Exact latent sequence posterior
#####


def latent_marginals(config: DiscreteHMMConfiguration, observation_sequence):
    init = int(config.linear_grid_dim / 2)
    initial_distribution = tfd.Categorical(logits=config.transition_tensor[init, :])
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


def latent_sequence_posterior(
    config: DiscreteHMMConfiguration, latent_point, observation_sequence
):
    hmm, _ = latent_marginals(config, observation_sequence)

    def _inner(carry, x):
        latent, obs = x
        v = jnp.log(carry[latent])
        v += jnp.log(jax.nn.softmax(hmm.observation_distribution.logits)[latent, obs])
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


class _DiscreteHMMLatentSequencePosterior(Distribution):
    def random_weighted(self, key, config, observation_sequence, **kwargs):
        key, sub_key = jax.random.split(key)
        _, (v, _) = forward_filtering_backward_sampling(
            sub_key, config, observation_sequence
        )
        key, (w, _) = self.estimate_logpdf(
            key, v, config, observation_sequence, **kwargs
        )
        return key, (w, v)

    def estimate_logpdf(self, key, v, config, observation_sequence, **kwargs):
        prob, _ = latent_sequence_posterior(config, v, observation_sequence)
        return key, (prob, v)

    def data_logpdf(self, config, observation_sequence):
        return log_data_marginal(config, observation_sequence)

    def get_forward_filters(self, key, config, observation_sequence):
        key, sub_key = jax.random.split(key)
        _, (_, ffs) = forward_filtering_backward_sampling(
            sub_key, config, observation_sequence
        )
        return key, ffs


##############
# Shorthands #
##############

DiscreteHMM = _DiscreteHMMLatentSequencePosterior()

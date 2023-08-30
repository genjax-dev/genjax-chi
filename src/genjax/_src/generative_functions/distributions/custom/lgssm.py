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
from genjax._src.generative_functions.distributions.distribution import ExactDensity

import jax
import jax.numpy as jnp

@dataclass
class LGSSM(ExactDensity):
    def sample_step(self, carry, zt):
        """
        Execute a single sampling step of the LGSSM.

        Args:
        - carry (tuple): A tuple containing the following:
            - key (jax.random.PRNGKey): JAX random key for generating noise.
            - x_t (float): The current state value.
            - alpha (float): Update rate for the state.
            - sigma_SE (float): Process noise.
            - sigma_noise (float): Measurement noise.

        - zt (float): The mean value for the measurement at the current time step.

        Returns:
        - new_carry (tuple): Updated state information after sampling.
        - (x_next, y_t) (tuple): The next state and the current measurement.
        """
        
        key, x_t, alpha, sigma_SE, sigma_noise = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # State update
        x_next = alpha * x_t + sigma_SE * jax.random.normal(subkey1)
        
        # Measurement update
        y_t = x_t + zt + sigma_noise * jax.random.normal(subkey2)
        
        new_carry = (key, x_next, alpha, sigma_SE, sigma_noise)
        
        return new_carry, (x_next, y_t)

    def sample(self, key, z, alpha, sigma_SE, sigma_noise, **kwargs):
        """Sample from the LGSSM for T time steps.
        
        Args:
        - key: A JAX random key.
        - z: Vector of mean values.
        - alpha: Update rate.
        - sigma_SE: Process noise.
        - sigma_noise: Measurement noise.
            
        Returns:
        - y: Measurement sequence.
        """
        initial_state = 0.0 # initial state mean

        carry = (key, initial_state, alpha, sigma_SE, sigma_noise)
        
        # Use lax.scan to efficiently loop over z values and produce state and measurement sequences
        _, (x_sequence, y_sequence) = jax.lax.scan(self.sample_step, carry, z)
        
        return y_sequence


    def kalman_step(self, carry, yt_zt):
        """
        Execute a single step of the Kalman filter for a given observation and mean value.

        Args:
        - carry (tuple): A tuple containing the following state information:
            - x_hat (float): The current estimated state.
            - P (float): The current state variance.
            - alpha (float): Update rate for the state estimation.
            - sigma_SE (float): Process noise.
            - sigma_noise (float): Measurement noise.
            
        - yt_zt (tuple): A tuple containing:
            - y_t (float): The measurement at the current time step.
            - z_t (float): The mean value for the measurement at the current time step.

        Returns:
        - new_carry (tuple): Updated state information after processing the current measurement.
        - log_likelihood_t (float): Log likelihood of the observation y_t given the current state and parameters.
        """
        
        y_t, z_t = yt_zt
        x_hat, P, alpha, sigma_SE, sigma_noise = carry
        
        # Prediction step
        x_hat_pred = alpha * x_hat
        P_pred = alpha**2 * P + sigma_SE**2
        
        # Measurement residual (innovation)
        innovation = y_t - z_t - x_hat_pred
        innovation_variance = P_pred + sigma_noise**2
        
        # Update log likelihood
        log_likelihood_t = -0.5 * (innovation**2 / innovation_variance + jnp.log(2 * jnp.pi * innovation_variance))
        
        # Update step
        K = P_pred / innovation_variance  # Kalman gain
        x_hat = x_hat_pred + K * innovation
        P = (1 - K) * P_pred
        
        new_carry = (x_hat, P, alpha, sigma_SE, sigma_noise)
        
        return new_carry, log_likelihood_t

    
    def logpdf(self, y, z, alpha, sigma_SE, sigma_noise, **kwargs):
        """Compute log probability of measurements given states and parameters.
        
        Args:
        - y: Measurement sequence.
        - alpha: Update rate.
        - sigma_SE: Process noise.
        - sigma_noise: Measurement noise.
        - z: Vector of mean values.
        
        Returns:
        - log_prob: Log probability of the measurements.
        """
        
        # Initial values
        x_hat = 0.0  # initial state mean
        P = sigma_SE**2  # initial state variance
        carry = (x_hat, P, alpha, sigma_SE, sigma_noise)
        
        # Use lax.scan to efficiently loop over measurements and z values
        _, log_likelihoods = jax.lax.scan(self.kalman_step, carry, (y, z))
        
        # Sum log likelihoods to get total log likelihood
        total_log_likelihood = jnp.sum(log_likelihoods)
        
        return total_log_likelihood

lgssm = LGSSM()
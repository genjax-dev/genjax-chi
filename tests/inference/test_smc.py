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

import genjax
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import logsumexp


class TestSMC:
    def test_exact_flip_flip_trivial(self):
        @genjax.static_gen_fn
        def flip_flip_trivial():
            _ = genjax.flip(0.5) @ "x"
            _ = genjax.flip(0.7) @ "y"

        def flip_flip_exact_log_marginal_density(target: genjax.Target):
            y = target["y"]
            return genjax.flip.logpdf(y, 0.7)

        key = jax.random.PRNGKey(314159)
        inference_problem = genjax.Target(
            flip_flip_trivial, (), genjax.choice_map({"y": True})
        )

        # Single sample IS.
        Z_est = genjax.inference.smc.Importance(
            inference_problem
        ).log_marginal_likelihood_estimate(key)
        Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
        assert Z_est == pytest.approx(Z_exact, 1e-1)

        # K-sample sample IS.
        Z_est = genjax.inference.smc.ImportanceK(
            inference_problem, k_particles=1000
        ).log_marginal_likelihood_estimate(key)
        Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
        assert Z_est == pytest.approx(Z_exact, 1e-3)

    def test_exact_flip_flip(self):
        @genjax.static_gen_fn
        def flip_flip():
            v1 = genjax.flip(0.5) @ "x"
            p = jax.lax.cond(v1, lambda: 0.9, lambda: 0.3)
            _ = genjax.flip(p) @ "y"

        def flip_flip_exact_log_marginal_density(target: genjax.Target):
            y = target["y"]
            x_prior = jnp.array(
                [
                    genjax.flip.logpdf(True, 0.5),
                    genjax.flip.logpdf(False, 0.5),
                ]
            )
            y_likelihood = jnp.array(
                [
                    genjax.flip.logpdf(y, 0.9),
                    genjax.flip.logpdf(y, 0.3),
                ]
            )
            y_marginal = logsumexp(x_prior + y_likelihood)
            return y_marginal

        key = jax.random.PRNGKey(314159)
        inference_problem = genjax.Target(flip_flip, (), genjax.choice_map({"y": True}))

        # K-sample IS.
        Z_est = genjax.inference.smc.ImportanceK(
            inference_problem, k_particles=2000
        ).log_marginal_likelihood_estimate(key)
        Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
        assert Z_est == pytest.approx(Z_exact, 1e-1)

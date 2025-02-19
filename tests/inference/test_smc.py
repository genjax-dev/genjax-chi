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
import pytest
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S
from genjax._src.core.typing import Any
from genjax._src.inference.sp import Target


def logpdf(v):
    return lambda c, *args: v.assess(C.v(c), args)[0]


class TestSMC:
    def test_exact_flip_flip_trivial(self):
        @genjax.gen
        def flip_flip_trivial():
            _ = genjax.flip(0.5) @ "x"
            _ = genjax.flip(0.7) @ "y"

        def flip_flip_exact_log_marginal_density(target: genjax.Target[Any]):
            y = target.constraint.get_submap("y")
            return genjax.flip.assess(y, (0.7,))[0]

        key = jax.random.key(314159)
        inference_problem = genjax.Target(flip_flip_trivial, (), C["y"].set(True))

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
        @genjax.gen
        def flip_flip():
            v1 = genjax.flip(0.5) @ "x"
            p = jax.lax.cond(v1, lambda: 0.9, lambda: 0.3)
            _ = genjax.flip(p) @ "y"

        def flip_flip_exact_log_marginal_density(target: genjax.Target[Any]):
            y = target["y"]
            x_prior = jnp.array([
                logpdf(genjax.flip)(True, 0.5),
                logpdf(genjax.flip)(False, 0.5),
            ])
            y_likelihood = jnp.array([
                logpdf(genjax.flip)(y, 0.9),
                logpdf(genjax.flip)(y, 0.3),
            ])
            y_marginal = logsumexp(x_prior + y_likelihood)
            return y_marginal

        key = jax.random.key(314159)
        inference_problem = genjax.Target(flip_flip, (), C["y"].set(True))

        # K-sample IS.
        Z_est = genjax.inference.smc.ImportanceK(
            inference_problem, k_particles=2000
        ).log_marginal_likelihood_estimate(key)
        Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
        assert Z_est == pytest.approx(Z_exact, 1e-1)

    def test_non_marginal_target(self):
        @genjax.gen
        def model():
            idx = genjax.categorical(probs=[0.5, 0.25, 0.25]) @ "idx"
            # under the prior, 50% chance to be in cluster 1 and 50% chance to be in cluster 2.
            means = jnp.array([0.0, 10.0, 11.0])
            vars = jnp.array([1.0, 1.0, 1.0])
            x = genjax.normal(means[idx], vars[idx]) @ "x"
            y = genjax.normal(means[idx], vars[idx]) @ "y"
            return x, y

        marginal_model = model.marginal(
            selection=S["x"] | S["y"]
        )  # This means we are projection onto the variables x and y, marginalizing out the rest

        obs1 = C["x"].set(1.0)
        with pytest.raises(TypeError):
            Target(marginal_model, (), obs1)

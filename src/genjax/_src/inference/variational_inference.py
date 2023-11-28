# Copyright 2023 MIT Probabilistic Computing Project
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
from jax.scipy.special import logsumexp
from optax import GradientTransformation
from optax import adam

from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.state import StateCombinator
from genjax._src.utilities import slash


@dataclass
class VariationalInference(Pytree):
    model: GenerativeFunction
    variational_model: StateCombinator
    gradient_samples_per_iter: IntArray
    optimizer: GradientTransformation
    optimizer_state: Any

    def flatten(self):
        return (self.optimizer, self.optimizer_state), (
            self.model,
            self.variational_model,
            self.gradient_samples_per_iter,
        )

    @typecheck
    @classmethod
    def new(
        cls,
        model: GenerativeFunction,
        variational_model: StateCombinator,
        iters: IntArray = 1000,
        gradient_samples_per_iter: IntArray = 100,
        optimizer: GradientTransformation = adam(1e-5),
    ):
        params = variational_model.get_params()
        opt_state = optimizer.init(params)
        return VariationalInference(
            model,
            variational_model,
            iters,
            gradient_samples_per_iter,
            optimizer,
            opt_state,
        )

    def _vimco_geometric_baselines(self, log_weights):
        num_samples = len(log_weights)
        s = jnp.sum(log_weights)
        log_weights = (s - log_weights) / (num_samples - 1)
        baselines = logsumexp(log_weights) - jnp.log(num_samples)
        return baselines

    def _multi_sample_gradient_estimate(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        var_model_args: Tuple,
        scale_factor: FloatArray = 1.0,
    ):
        key, sub_keys = slash(key, self.gradient_samples_per_iter)
        sub_keys, var_trace = jax.vmap(
            self.variational_model.simulate, in_axes=(0, None)
        )(sub_keys, var_model_args)

        def _assess(key, var_trace):
            var_choices = var_trace.strip()
            constraints = observations.safe_merge(var_choices)
            key, (_, model_weight) = self.model.assess(key, constraints, model_args)
            return key, model_weight

        sub_keys, log_weights = jax.vmap(_assess, in_axes=(0, 0))(sub_keys, var_trace)

        log_total_weight = logsumexp(log_weights)
        L = log_total_weight - jnp.log(self.gradient_samples_per_iter)
        baselines = self._vimco_geometric_baselines(log_weights)
        weights_normalized = jnp.exp(log_weights - log_total_weight)
        learning_signal = (L - baselines) - weights_normalized
        params = self.variational_model.get_params()

        def _score(key, params):
            key, logpdf = jax.vmap(
                self.variational_model.score_params, in_axes=(0, 0, None)
            )(key, var_trace, params)
            return logpdf

        params_grad = jax.grad(_score, argnums=1)(sub_keys, params)
        params_grad = jtu.tree_map(
            lambda v: v * learning_signal * scale_factor, params_grad
        )

        return key, (params_grad, L)

    def _vimco_apply(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        var_model_args: Tuple,
    ):
        key, (params_grad, L) = self._multi_sample_gradient_estimate(
            key,
            observations,
            model_args,
            var_model_args,
            1.0 / self.gradient_samples_per_iter,
        )
        iwelbo_estimate = L / self.gradient_samples_per_iter
        updates, self.opt_state = self.optimizer.update(params_grad, self.opt_state)
        self.variational_model.update_params(updates)
        return key, iwelbo_estimate

    @typecheck
    def __call__(
        self,
        key: PRNGKey,
        observations: ChoiceMap,
        model_args: Tuple,
        var_model_args: Tuple,
    ):
        return self._vimco_apply(key, observations, model_args, var_model_args)


##############
# Shorthands #
##############

variational = VariationalInference.new

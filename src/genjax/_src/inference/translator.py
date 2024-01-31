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

import abc

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.checkify import check

from genjax._src.checkify import optional_check
from genjax._src.core.datatypes.generative import (
    Choice,
    ChoiceMap,
    GenerativeFunction,
    Trace,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Bool,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)

#####################
# Trace translators #
#####################


class TraceTranslator(Pytree):
    @abc.abstractmethod
    def apply(
        self,
        key: PRNGKey,
        prev_model_trace: Trace,
    ) -> Tuple[Trace, FloatArray]:
        raise NotImplementedError

    @typecheck
    def __call__(
        self,
        key: PRNGKey,
        prev_model_trace: Trace,
    ) -> Tuple[Trace, FloatArray]:
        return self.apply(key, prev_model_trace)


############################
# Jacobian det array stack #
############################


def stack_differentiable(v):
    grad_tree, _ = Pytree.tree_grad_split(v)
    leaves = jtu.tree_leaves(grad_tree)
    stacked = jnp.stack(leaves) if len(leaves) > 1 else leaves[0]
    return stacked


def safe_slogdet(v):
    if v.shape == ():
        return jnp.linalg.slogdet(jnp.array([[v]], copy=False))
    else:
        return jnp.linalg.slogdet(v)


#####################################
# Simple extending trace translator #
#####################################


class ExtendingTraceTranslator(TraceTranslator):
    choice_map_forward: Callable = Pytree.static()  # part of bijection.
    choice_map_inverse: Callable = Pytree.static()  # part of bijection
    check_bijection: Bool
    p_argdiffs: Tuple
    q_forward: GenerativeFunction
    q_forward_args: Tuple
    new_observations: Choice

    def value_and_jacobian_correction(self, forward, trace):
        trace_choices = trace.get_choice()
        grad_tree, no_grad_tree = Pytree.tree_grad_split(trace_choices)

        def _inner(differentiable):
            choices = Pytree.tree_grad_zip(differentiable, no_grad_tree)
            out_choices = forward(choices)
            return out_choices, out_choices

        inner_jacfwd = jax.jacfwd(_inner, has_aux=True)
        J, transformed = inner_jacfwd(grad_tree)
        if self.check_bijection:

            def optional_check_bijection_is_bijection():
                backwards = self.choice_map_inverse(transformed)
                flattened = jtu.tree_leaves(
                    jtu.tree_map(
                        lambda v1, v2: jnp.all(v1 == v2),
                        trace_choices,
                        backwards,
                    )
                )
                check_flag = jnp.all(jnp.array(flattened))
                check(check_flag, "Bijection check failed")

            optional_check(optional_check_bijection_is_bijection)
        J = stack_differentiable(J)
        (_, J_log_abs_det) = safe_slogdet(J)
        return transformed, J_log_abs_det

    def apply(self, key: PRNGKey, prev_model_trace: Trace):
        prev_model_choices = prev_model_trace.get_choice()
        forward_proposal_trace = self.q_forward.simulate(
            key, (self.new_observations, prev_model_choices, *self.q_forward_args)
        )
        transformed, J_log_abs_det = self.value_and_jacobian_correction(
            self.choice_map_forward, forward_proposal_trace
        )
        forward_proposal_score = forward_proposal_trace.get_score()
        constraints = transformed.merge(self.new_observations)
        (new_model_trace, log_model_weight, _, discard) = prev_model_trace.update(
            key, constraints, self.p_argdiffs
        )
        # This type of trace translator does not handle proposing
        # to existing latents.
        assert discard.is_empty()
        log_weight = log_model_weight - forward_proposal_score - J_log_abs_det
        return (new_model_trace, log_weight)


def extending_trace_translator(
    p_argdiffs: Tuple,
    q_forward: GenerativeFunction,
    q_forward_args: Tuple,
    new_obs: ChoiceMap,
    choice_map_forward: Callable = lambda v: v,
    choice_map_backward: Callable = lambda v: v,
    check_bijection=False,
):
    return ExtendingTraceTranslator(
        choice_map_forward,
        choice_map_backward,
        check_bijection,
        p_argdiffs,
        q_forward,
        q_forward_args,
        new_obs,
    )


###########################
# Trace kernels for SMCP3 #
###########################

# Note that this class of trace translators is strictly more general
# than `ExtendingTraceTranslator`, but more advanced to use
# (a power tool).


class TraceKernelTraceTranslator(TraceTranslator):
    """
    A trace translator for expressing SMCP³ moves (c.f. [SMCP³: Sequential Monte Carlo with Probabilistic Program Proposals](https://proceedings.mlr.press/v206/lew23a/lew23a.pdf)).

    Requires that users specify K (forward) and L (backward) probabilistic program kernels using the `genjax.static_gen_fn` language.

    The K kernel should return a choice map of new choices to perform the update move with (`x_new`). It may also sample auxiliary randomness ('aux') to construct these new choices. It represents the distribution P(x_new, aux | x).

    The L kernel should explicitly trace any auxiliary randomness ('aux') sampled in K. It is used to improve the weight quality - by attempting to invert the (potentially stochastic) logic of K to produce ('aux'). It represents the distribution P(aux | x_new)

    Then, E_aux[P(aux | x_new) / P(x_new, aux | x)] = E_aux[1 / w] = 1 / P(x_new), allowing us to use the proposal from K as a valid proposal with the weight contribution of 1 / w.
    """

    model_argdiffs: Tuple
    K: StaticGenerativeFunction
    K_args: Tuple
    L: StaticGenerativeFunction
    L_args: Tuple

    def value_and_jacobian_correction(
        self,
        prev_model_choices,
        K_aux_choices,
    ):
        grad_tree, no_grad_tree = Pytree.tree_grad_split(prev_model_choices)

        def _inner(differentiable):
            prev_model_choices = Pytree.tree_grad_zip(differentiable, no_grad_tree)
            (_, new_choices) = self.K.assess(
                K_aux_choices, (prev_model_choices, *self.K_args)
            )
            return new_choices

        inner_jacfwd = jax.jacfwd(_inner)
        J = inner_jacfwd(grad_tree)
        (_, J_log_abs_det) = safe_slogdet(J)
        return J_log_abs_det

    @typecheck
    def apply(
        self,
        key: PRNGKey,
        prev_model_trace: Trace,
    ) -> Tuple[Trace, FloatArray]:
        key, sub_key = jax.random.split(key)
        prev_model_choices = prev_model_trace.get_choice()
        aux_choices, K_score, new_choices = self.K.propose(
            sub_key, (prev_model_choices, self.K_args)
        )
        assert isinstance(new_choices, Choice)
        J_log_abs_det = self.value_and_jacobian_correction(
            aux_choices, prev_model_choices
        )
        (new_model_trace, update_weight, _, discard) = prev_model_trace.update(
            key, new_choices, self.model_argdiffs
        )
        assert discard.is_empty()
        new_model_choices = new_model_trace.get_choice()
        L_score = self.L.assess(
            aux_choices,
            (new_model_choices, self.L_args),
        )
        weight = update_weight + (L_score - K_score) + J_log_abs_det
        return new_model_trace, weight

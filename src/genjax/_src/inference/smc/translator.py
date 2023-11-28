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
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from genjax._src.core.datatypes.generative import Choice, ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)


#####################
# Trace translators #
#####################


@dataclass
class TraceTranslator(Pytree):
    @abc.abstractmethod
    def inverse(self, prev_model_trace, prev_observations):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self,
        key: PRNGKey,
        prev_model_trace: Trace,
    ) -> Tuple[Trace, FloatArray]:
        raise NotImplementedError


@dataclass
class DeterministicTraceTranslator(TraceTranslator):
    forward: Callable  # bijection forward
    backward: Callable  # bijection inverse
    p_new: GenerativeFunction
    p_args: Tuple
    new_obs: Choice

    def flatten(self):
        return (
            self.p_new,
            self.p_args,
            self.new_obs,
        ), (self.forward, self.backward)

    @classmethod
    def new(
        cls,
        p_new: GenerativeFunction,
        p_args: Tuple,
        new_obs: Choice,
        forward: Callable,
        backward: Callable,
    ):
        return DeterministicTraceTranslator(forward, backward, p_new, p_args, new_obs)

    def value_and_jacobian_correction(self, forward, trace):
        grad_tree, no_grad_tree = tree_grad_split(trace.get_choices())

        def _inner(differentiable):
            choices = tree_zipper(differentiable, no_grad_tree)
            out_choices = forward(choices)
            return out_choices

        inner_jacfwd = jax.jacfwd(_inner)
        J = inner_jacfwd(grad_tree)
        return jnp.slogdet(J)

    def run_transform(self, key: PRNGKey, prev_model_trace: Trace):
        results, log_abs_det = self.value_and_jacobian_correction(
            self.forward, prev_model_trace
        )
        constraints = results.merge(self.new_obs)
        new_trace, _ = self.p_new.importance(key, constraints, self.p_args)
        return new_trace, log_abs_det

    @typecheck
    def apply(
        self,
        key: PRNGKey,
        prev_model_trace: Trace,
    ) -> Tuple[Trace, FloatArray]:
        new_model_trace, log_abs_det = self.run_transform(key, prev_model_trace)
        prev_model_score = prev_model_trace.get_score()
        new_model_score = new_model_trace.get_score()
        log_weight = new_model_score - prev_model_score + log_abs_det
        return new_model_trace, log_weight


@typecheck
def deterministic_trace_translator(
    p_new: GenerativeFunction,
    p_args: Tuple,
    new_obs: Choice,
    forward: Callable,
    backward: Callable,
):
    return DeterministicTraceTranslator.new(p_new, p_args, new_obs, forward, backward)


#####################################
# Simple extending trace translator #
#####################################


@dataclass
class SimpleExtendingTraceTranslator(TraceTranslator):
    choice_map_bijection: Callable
    p_argdiffs: Tuple
    q_forward: GenerativeFunction
    q_forward_args: Tuple
    new_observations: Choice

    def flatten(self):
        return (
            self.p_argdiffs,
            self.q_forward,
            self.q_forward_args,
            self.new_observations,
        ), (self.choice_map_bijection,)

    @classmethod
    def new(
        cls,
        p_argdiffs: Tuple,
        q_forward: GenerativeFunction,
        q_forward_args: Tuple,
        new_obs: Choice,
        choice_map_bijection: Callable,
    ):
        return SimpleExtendingTraceTranslator(
            choice_map_bijection,
            p_argdiffs,
            q_forward,
            q_forward_args,
            new_obs,
        )

    def apply(self, key: PRNGKey, prev_model_trace: Trace):
        prev_model_choices = prev_model_trace.get_choices()
        forward_proposal_trace = self.q_forward.simulate(
            key, (prev_model_choices, *self.q_forward_args)
        )
        forward_proposal_score = forward_proposal_trace.get_score()
        constraints = forward_proposal_trace.get_choices().merge(self.new_obs)
        (new_model_trace, log_model_weight, _, discard) = prev_model_trace.update(
            key, constraints, self.p_argdiffs
        )
        assert discard.is_empty()
        log_weight = log_model_weight - forward_proposal_score
        return (new_model_trace, log_weight)


@typecheck
def simple_extending_trace_translator(
    p_argdiffs: Tuple,
    q_forward: GenerativeFunction,
    q_forward_args: Tuple,
    new_obs: ChoiceMap,
    choice_map_bijection: Callable,
):
    return SimpleExtendingTraceTranslator.new(
        p_argdiffs,
        q_forward,
        q_forward_args,
        new_obs,
        choice_map_bijection,
    )


###########################
# Trace kernels for SMCP3 #
###########################


@dataclass
class TraceKernel(Pytree):
    forward: StaticGenerativeFunction
    backward: StaticGenerativeFunction

    def flatten(self):
        return (self.forward, self.backward), ()

    def assess(self, key, choices, args):
        (r, w) = self.gen_fn.assess(key, choices, args)
        (chm, aux) = r
        return ((chm, aux), w)

    def propose(self, key, args):
        tr = self.gen_fn.simulate(key, args)
        choices = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        (chm, aux) = retval
        return (choices, score, (chm, aux))

    def jacfwd_retval(self, key, choices, trace, proposal_args):
        grad_tree, no_grad_tree = tree_grad_split((choices, trace))

        def _inner(differentiable: Tuple):
            choices, trace = tree_zipper(differentiable, no_grad_tree)
            ((chm, aux), _) = self.gen_fn.assess(key, choices, (trace, proposal_args))
            return (chm, aux)

        inner_jacfwd = jax.jacfwd(_inner)
        J = inner_jacfwd(grad_tree)
        return J


######################
# Language decorator #
######################


@typecheck
def trace_kernel(gen_fn: StaticGenerativeFunction):
    return TraceKernel.new(gen_fn)

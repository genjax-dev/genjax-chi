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
import jax.tree_util as jtu
from jax.experimental.checkify import check

from genjax._src.checkify import optional_check
from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.typing import Bool
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
    grad_tree, _ = tree_grad_split(v)
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


@dataclass
class ExtendingTraceTranslator(TraceTranslator):
    choice_map_forward: Callable  # part of bijection
    choice_map_inverse: Callable  # part of bijection
    check_bijection: Bool
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
        ), (self.choice_map_forward, self.choice_map_inverse, self.check_bijection)

    @classmethod
    def new(
        cls,
        p_argdiffs: Tuple,
        q_forward: GenerativeFunction,
        q_forward_args: Tuple,
        new_obs: Choice,
        choice_map_forward: Callable,
        choice_map_inverse: Callable,
        check_bijection: Bool,
    ):
        return ExtendingTraceTranslator(
            choice_map_forward,
            choice_map_inverse,
            check_bijection,
            p_argdiffs,
            q_forward,
            q_forward_args,
            new_obs,
        )

    def value_and_jacobian_correction(self, forward, trace):
        trace_choices = trace.get_choices()
        grad_tree, no_grad_tree = tree_grad_split(trace_choices)

        def _inner(differentiable):
            choices = tree_zipper(differentiable, no_grad_tree)
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
        prev_model_choices = prev_model_trace.get_choices()
        forward_proposal_trace = self.q_forward.simulate(
            key, (self.new_observations, prev_model_choices, *self.q_forward_args)
        )
        transformed, log_abs_det = self.value_and_jacobian_correction(
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
        log_weight = log_model_weight - forward_proposal_score - log_abs_det
        return (new_model_trace, log_weight)


@typecheck
def extending_trace_translator(
    p_argdiffs: Tuple,
    q_forward: GenerativeFunction,
    q_forward_args: Tuple,
    new_obs: ChoiceMap,
    choice_map_forward: Callable,
    choice_map_backward: Callable,
    check_bijection=False,
):
    return ExtendingTraceTranslator.new(
        p_argdiffs,
        q_forward,
        q_forward_args,
        new_obs,
        choice_map_forward,
        choice_map_backward,
        check_bijection,
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

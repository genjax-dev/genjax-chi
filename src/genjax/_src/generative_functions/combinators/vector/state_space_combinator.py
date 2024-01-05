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

import functools
from dataclasses import dataclass
from typing import Callable

from genjax._src.core.datatypes.generative import ChoiceMap, GenerativeFunction, Trace
from genjax._src.core.typing import (
    Any,
    FloatArray,
    IntArray,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.combinators.vector.unfold_combinator import Unfold


@dataclass
class StateSpaceChoiceMap(ChoiceMap):
    initial: ChoiceMap
    transition: ChoiceMap


@dataclass
class StateSpaceTrace(Trace):
    state_space: GenerativeFunction
    initial_trace: Trace
    transition_trace: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.state_space,
            self.initial_trace,
            self.transition_trace,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return StateSpaceChoiceMap.new(
            self.initial_trace,
            self.transition_trace,
        )

    def get_gen_fn(self):
        return self.state_space

    def get_score(self):
        return self.score

    def get_retval(self):
        return self.retval

    def project(self, selection):
        pass


@dataclass
class StateSpaceCombinator(GenerativeFunction):
    max_length: IntArray
    initial_model: GenerativeFunction
    transition_model: GenerativeFunction

    def flatten(self):
        return (self.initial_model, self.transition_model), (self.max_length,)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[PRNGKey, StateSpaceTrace]:
        initial_model_args, (dynamic_length, *static_args) = args
        key, initial_trace = self.initial_model.simulate(key, initial_model_args)
        initial_retval = initial_trace.get_retval()
        chain = Unfold(self.transition_model, max_length=self.max_length)
        key, transition_trace = chain.simulate(
            key, (dynamic_length, initial_retval, *static_args)
        )
        chain_retval = transition_trace.get_retval()
        score = initial_trace.get_score() + transition_trace.get_score()
        state_space_trace = StateSpaceTrace.new(
            self,
            initial_trace,
            transition_trace,
            args,
            (initial_retval, chain_retval),
            score,
        )
        return key, state_space_trace

    def importance(self, *args):
        pass

    def update(self, *args):
        pass

    def assess(self, *args):
        pass


#############
# Decorator #
#############


def StateSpace(
    *, max_length, initial_model, transition_model
) -> Callable[[Callable], StateSpaceCombinator]:
    def decorator(f) -> StateSpaceCombinator:
        gf = StateSpaceCombinator(
            max_length=max_length,
            initial_model=initial_model,
            transition_model=transition_model,
        )
        functools.update_wrapper(gf, f)
        return gf

    return decorator

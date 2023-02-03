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

"""This module provides a combinator which transforms a generative function
into a :code:`nn.Module`-like object that holds learnable parameters.

It exposes an extended set of interfaces (new: :code:`param_grad` and :code:`update_state`) which allow programmatic computation of gradients with respect to held parameters, as well as updating parameters.

It enables learning idioms which cohere with other packages in the JAX ecosystem (e.g. supporting :code:`optax` optimizers).
"""

from dataclasses import dataclass
from typing import Any

import jax.tree_util as jtu

from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import GenerativeFunction


@dataclass
class LearnCombinator(GenerativeFunction):
    inner: GenerativeFunction
    state: Any

    def flatten(self):
        return (self.inner, self.state), ()

    def simulate(self, key, args):
        return self.inner.simulate(key, (*args[0:-1], self.state))

    def importance(self, key, chm, args):
        return self.inner.importance(
            key,
            chm,
            (*args[0:-1], self.state),
        )

    def update(self, key, prev, chm, args):
        return self.inner.update(
            key,
            prev,
            chm,
            (*args[0:-1], self.state),
        )

    def assess(self, key, chm, args):
        return self.inner.assess(
            key,
            chm,
            (*args[0:-1], self.state),
        )

    def score_state(self, key, tr, state):
        gen_fn = tr.get_gen_fn()
        choices = tr.strip()
        args = tr.get_args()
        key, scorer, _ = gen_fn.unzip(key, choices)
        logpdf = scorer(EmptyChoiceMap(), (*args[0:-1], state))
        return key, logpdf

    def update_state(self, updates):
        def _apply_update(u, p):
            if u is None:
                return p
            else:
                return p + u

        def _is_none(x):
            return x is None

        self.state = jtu.tree_map(
            _apply_update,
            updates,
            self.state,
            is_leaf=_is_none,
        )

    def get_state(self):
        return self.state


##############
# Shorthands #
##############

Learn = LearnCombinator.new

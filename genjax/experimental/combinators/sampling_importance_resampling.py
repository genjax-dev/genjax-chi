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

import jax
import jax.tree_util as jtu
import numpy as np
from genjax.core.datatypes import GenerativeFunction, Trace
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class SIRTrace(Trace):
    gen_fn: SIRCombinator
    args: Tuple
    inner: Trace
    retval: Any
    score: Any

    def flatten(self):
        return (self.args, self.inner, self.retval, self.score), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return SIRTrace(*data, *xs)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.inner.get_choices()


@dataclass
class SIRCombinator(GenerativeFunction):
    model: GenerativeFunction
    proposal: GenerativeFunction
    n_particles: int

    def flatten(self):
        return (), (self.model, self.proposal, self.n_particles)

    @classmethod
    def unflatten(cls, data, xs):
        return SIRCombinator(*data, *xs)

    def simulate(self, key, args):
        (observations, model_args, proposal_args) = args
        key, ptr = jax.vmap(self.proposal.simulate, in_axes=(0, (None, None)))(
            key, (observations, proposal_args)
        )
        obs_none_tree = jtu.tree_map(lambda v: None, observations)
        ptr_none_tree = jtu.tree_map(lambda v: None, ptr)
        none_tree = ptr_none_tree.merge(obs_none_tree)
        constraints = ptr.merge(observations)
        key, (lmws, mtr) = jax.vmap(
            self.model.importance, in_axes=(0, None, None)
        )(key, constraints, model_args)
        lws = lmws - ptr.get_score()
        ltw = jax.scipy.special.logsumexp(lws)
        lnw = lws - ltw
        lmle = ltw - np.log(self.n_particles)
        selected = jax.random.categorical(key, lnw)
        ptr = jtu.tree_map(lambda v: v[selected], ptr)
        retval = mtr.get_retval()[selected]
        score = lmws[selected] - lmle
        return key, SIRTrace(self, args, ptr, retval, score)

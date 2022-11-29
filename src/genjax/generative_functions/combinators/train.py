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

It exposes an extended set of interfaces (new: :code:`param_grad` and :code:`update_params`) which allow programmatic computation of gradients with respect to held parameters, as well as updating parameters.

It enables learning idioms which cohere with other packages in the JAX ecosystem (e.g. supporting :code:`optax` optimizers).
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.tree_util as jtu

from genjax.core.datatypes import GenerativeFunction


@dataclass
class TrainCombinator(GenerativeFunction):
    inner: GenerativeFunction
    params: Any

    def flatten(self):
        return (self.params), (self.inner,)

    @classmethod
    def unflatten(self, xs, data):
        return TrainCombinator(*xs, *data)

    def __call__(self, key, *args, **kwargs):
        return self.inner.__call__(
            key,
            *args,
            self.params,
            **kwargs,
        )

    def simulate(self, key, args):
        return self.inner.simulate(key, (*args, self.params))

    def importance(self, key, chm, args):
        return self.inner.importance(
            key,
            chm,
            (*args, self.params),
        )

    def update(self, key, prev, chm, args):
        return self.inner.update(
            key,
            prev,
            chm,
            (*args, self.params),
        )

    def param_grad(self, key, tr, scale=1.0):
        def _inner(key, params, tr):
            chm = tr.get_choices()
            args = tr.get_args()
            key, (w, tr) = self.inner.importance(
                key,
                chm,
                (*args[0:-1], params),
            )
            return tr.get_score(), key

        params = self.params
        grad, key = jax.grad(_inner, argnums=1, has_aux=True)(key, params, tr)
        grad = jtu.tree_map(lambda v: v * scale, grad)
        return key, grad

    def update_params(self, updates):
        def _apply_update(u, p):
            if u is None:
                return p
            else:
                return p + u

        def _is_none(x):
            return x is None

        self.params = jtu.tree_map(
            _apply_update,
            updates,
            self.params,
            is_leaf=_is_none,
        )

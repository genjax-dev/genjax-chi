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

"""
This module provides a combinator which transforms a generative function into a :code:`nn.Module`-like object that holds learnable parameters.

It exposes an extended set of interfaces (new: :code:`param_grad`
and :code:`update_params`) which allow programmatic computation of
gradients with respect to held parameters, as well as updating parameters.

It enables learning idioms which cohere with other packages
in the JAX ecosystem (e.g. supporting :code:`optax` optimizers).
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.tree_util as jtu

from genjax.core.datatypes import GenerativeFunction


@dataclass
class TrainableCombinator(GenerativeFunction):
    """
    Example
    -------

    .. jupyter-execute::

        import jax
        import genjax
        import optax

        # Here, we can pass params as a keyword argument to
        # genjax.Trainable.
        #
        # The combinator expects a generative function which accepts the params
        # argument at the last argument position.
        @genjax.gen(
            genjax.Trainable,
            params={"x": 0.5},
        )
        def model(key, params):
            x = params["x"]
            key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.5))
            return key, y


        def learning(key, lr, chm):
            optim = optax.adam(lr)
            opt_state = optim.init(model.params)
            for _ in range(0, 20):
                key, (w, tr) = genjax.importance(model)(key, chm, ())

                # Usage here.
                key, grad = model.param_grad(key, tr, scale=w)
                updates, opt_state = optim.update(grad, opt_state)
                model.update_params(updates)
            return model.params


        key = jax.random.PRNGKey(314159)
        learning_rate = 3e-3
        obs = genjax.ChoiceMap.new({("y",): 0.2})
        trained = jax.jit(learning)(key, learning_rate, obs)
    """

    inner: GenerativeFunction
    params: Any

    def flatten(self):
        return (self.params), (self.inner,)

    @classmethod
    def unflatten(self, xs, data):
        return TrainableCombinator(*xs, *data)

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

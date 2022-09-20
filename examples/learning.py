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
import genjax
import optax


@genjax.gen(
    genjax.Learn,
    params={"x": 0.5},
)
def model(key, params):
    x = params["x"]
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.5))
    return key, y


def learning(key, lr, chm):
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model.params)
    for _ in range(0, 100):
        key, (w, tr) = genjax.importance(model)(key, chm, ())
        key, grad = model.param_grad(key, tr, scale=w)
        updates, opt_state = optim.update(grad, opt_state)
        model.update_params(updates)
    return model.params


key = jax.random.PRNGKey(314159)
learning_rate = 3e-3
obs = genjax.ChoiceMap.new({("y",): 0.2})
trained = jax.jit(learning)(key, learning_rate, obs)
print(trained)

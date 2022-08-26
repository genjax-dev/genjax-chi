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


@genjax.gen
def minimal_model(rng, x, sigma):
    rng, a = genjax.trace("a", genjax.Normal)(rng, ())
    rng, b = genjax.trace("b", genjax.Normal)(rng, ())
    y_clean = a * x + b
    rng, eps = genjax.trace("eps", genjax.Normal)(rng, ())
    y = y_clean + eps
    return rng, y


@genjax.gen
def hierarchical_model(rng, xs):
    rng, sigma = genjax.trace("sigma", genjax.Uniform)(rng, (0.1, 2))

    for i in range(len(xs)):
        x = xs[i]
        rng, y = genjax.trace(str(i), minimal_model)(rng, (x, sigma))

    return rng


def minimal_inference():
    rng = jax.random.PRNGKey(0)

    # trace one application of the function, to one datapoint
    # rng, trace_minimal = genjax.simulate(minimal_model)(rng, (0.0, 1.0,))

    xs = (1.0, 2.0)
    rng, trace_hierarchical = genjax.simulate(hierarchical_model)(rng, (xs,))


minimal_inference()

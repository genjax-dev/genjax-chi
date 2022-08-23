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


# This showcases a simple Metropolis-Hastings benchmark.


@genjax.gen
def normal_model(key):
    x = genjax.trace("x", genjax.Normal)(key)
    return x


@genjax.gen
def uniform_proposal(key, tr, d):
    current = tr["x"].get_value()
    key, x = genjax.trace("x", genjax.Uniform)(key, current - d, current + d)
    return key, x


# Simulate from the model.
key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(normal_model))(key, ())

# Inference with proposal MH.
inf = jax.jit(genjax.metropolis_hastings(normal_model, uniform_proposal))


def run_inference(key, tr):
    for _ in range(2, 10):
        key, (tr, _) = inf(key, tr, (0.25,))
    return (key, tr)


def test_simple_mh_benchmark(benchmark):
    key = jax.random.PRNGKey(314159)
    key, tr = jax.jit(genjax.simulate(normal_model))(key, ())
    jitted = jax.jit(run_inference)
    new_key, trace = benchmark(jitted, key, tr)

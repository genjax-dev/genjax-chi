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
import tensorflow as tf
from jax.experimental import jax2tf

import genjax


@genjax.gen
def model(key):
    x = genjax.trace("x", genjax.Normal)(key, ())
    return x


def __inner(key, args):
    key, tr = genjax.simulate(model)(key, args)
    return key, tr.get_score()


key = jax.random.PRNGKey(314159)
f_tf = jax2tf.convert(__inner)
key, score = f_tf(key, ())
print(score)

my_model = tf.Module()
my_model.f = tf.function(f_tf, autograph=False, jit_compile=True)
key, score = my_model.f(key, ())

tf.saved_model.save(
    my_model,
    ".",
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
)

restored_model = tf.saved_model.load(".")

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

import io

import genjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from genjax.serialization import msgpack_serialize


def eq(tr, other):
    return jtu.tree_all(jtu.tree_map(lambda v1, v2: jnp.all(v1 == v2), tr, other))


class TestMsgPackSerialize:
    def test_msgpack_round_trip(self):
        @genjax.gen
        def model(p):
            x = genjax.flip(p) @ "x"
            return x

        key = jax.random.PRNGKey(0)
        tr = model.simulate(key, (0.5,))
        bytes = msgpack_serialize.serialize(tr)

        restored_tr = msgpack_serialize.deserialize(bytes, model, (0.5,))
        assert eq(restored_tr, tr)

        # Test round-trip
        @genjax.gen
        def model_copy(p):
            x = genjax.flip(p) @ "x"
            return x

        restored_tr = msgpack_serialize.deserialize(bytes, model_copy, (0.5,))

        expected, _ = jax.tree_util.tree_flatten(tr)
        soln, _ = jax.tree_util.tree_flatten(restored_tr)
        for e, s in zip(expected, soln):
            assert e == s

    def test_msgpack_tensors(self):
        @genjax.gen
        def model(obs):
            x = genjax.flip(jnp.sum(obs) / len(obs)) @ "x"
            return x

        key = jax.random.PRNGKey(0)
        tr = model.simulate(
            key,
            (
                jnp.array(
                    [
                        1.0,
                        2.0,
                        3.0,
                    ]
                ),
            ),
        )
        bytes = msgpack_serialize.serialize(tr)

        restored_tr = msgpack_serialize.deserialize(
            bytes,
            model,
            (
                jnp.array(
                    [
                        1.0,
                        2.0,
                        3.0,
                    ]
                ),
            ),
        )
        assert eq(tr, restored_tr)

    def test_msgpack_auxilliary_methods(self):
        @genjax.gen
        def model(p):
            x = genjax.flip(p) @ "x"
            return x

        key = jax.random.PRNGKey(0)
        tr = model.simulate(key, (0.5,))

        # Raw bytes
        assert eq(
            tr,
            msgpack_serialize.deserialize(msgpack_serialize.dumps(tr), model, (0.5,)),
        )

        # IO
        f = io.BytesIO()
        msgpack_serialize.dump(tr, f)
        f.seek(0)
        assert eq(tr, msgpack_serialize.load(f, model, (0.5,)))

    def test_msgpack_default_extras(self):
        @genjax.gen
        def model():
            return jnp.array([1, 2, 3], dtype=jnp.bfloat16)

        key = jax.random.PRNGKey(0)
        tr = model.simulate(key, ())
        bytes = msgpack_serialize.serialize(tr)

        restored_tr = msgpack_serialize.deserialize(bytes, model, ())
        assert eq(tr, restored_tr)

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

import genjax
import jax
from genjax._src.core.serialization.msgpack import msgpack_serialize

class TestMsgPackSerialize:
    def test_serialize_round_trip(self):
        @genjax.static_gen_fn
        def model(p):
            x = genjax.flip(p) @ "x"
            return x
        
        key = jax.random.PRNGKey(0)
        tr = model.simulate(key, (0.5,))
        bytes = msgpack_serialize.serialize(tr)

        restored_tr = msgpack_serialize.deserialize(bytes, model)
        assert restored_tr == tr

        def model_copy(p):
            x = genjax.flip(p) @ "x"
            return x
        
        restored_tr = msgpack_serialize.deserialize(bytes, model_copy)
        # cannot compare generative functions
        assert jax.tree_util.tree_flatten(tr) == jax.tree_util.tree_flatten(restored_tr) 
    
    def test_serialize_tensors(self):
        @genjax.static_gen_fn
        def model(obs):
            x = genjax.flip(jax.sum(obs) / len(obs)) @ "x"
            return x

        key = jax.random.PRNGKey(0)
        tr = model.simulate(key, (jax.Array([1.0,2.0,3.0,]),))
        bytes = msgpack_serialize.serialize(tr)

        restored_tr = msgpack_serialize.deserialize(bytes, model)
        assert restored_tr.args == tr.args
        assert restored_tr == tr
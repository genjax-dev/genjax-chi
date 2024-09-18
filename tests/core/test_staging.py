# Copyright 2024 MIT Probabilistic Computing Project
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

import jax.numpy as jnp
from jax.random import PRNGKey

from genjax import ChoiceMap as Chm
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal
from genjax._src.core.interpreters.staging import FlagOp
from genjax.core.interpreters import (
    get_importance_shape,
    get_update_shape,
)


class TestStaging:
    def test_static_importance_shape(self):
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x

        tr, _ = get_importance_shape(model, C.n(), ())
        assert isinstance(tr.get_sample(), Chm)

    def test_static_update_shape(self):
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x

        key = PRNGKey(0)
        trace = model.simulate(key, ())
        new_trace, _w, _rd, bwd_request = get_update_shape(model, trace, C.n(), ())
        assert isinstance(new_trace.get_sample(), Chm)
        assert isinstance(bwd_request, Chm)


class TestFlag:
    def test_basic_operation(self):
        true_flags = [
            True,
            jnp.array(True),
            jnp.array([True, True]),
        ]
        false_flags = [
            False,
            jnp.array(False),
            jnp.array([False, False]),
        ]
        for t in true_flags:
            assert jnp.all(t)
            assert not jnp.all(FlagOp.not_(t))
            for f in false_flags:
                assert not jnp.all(f)
                assert not jnp.all(FlagOp.and_(t, f))
                assert jnp.all(FlagOp.or_(t, f))
                assert jnp.all(FlagOp.xor_(t, f))
            for u in true_flags:
                assert jnp.all(FlagOp.and_(t, u))
                assert jnp.all(FlagOp.or_(t, u))
                assert not jnp.all(FlagOp.xor_(t, u))
        for f1 in false_flags:
            for f2 in false_flags:
                assert not jnp.all(FlagOp.xor_(f1, f2))

    def test_where(self):
        assert FlagOp.where(True, 3.0, 4.0) == 3
        assert FlagOp.where(False, 3.0, 4.0) == 4
        assert FlagOp.where(jnp.array(True), 3.0, 4.0) == 3
        assert FlagOp.where(jnp.array(False), 3.0, 4.0) == 4

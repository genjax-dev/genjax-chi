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
from genjax import UpdateProblemBuilder as U
from genjax import gen, normal
from genjax._src.core.interpreters.staging import Flag
from genjax.core.interpreters import get_importance_shape, get_update_shape


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
        new_trace, _w, _rd, bwd_problem = get_update_shape(model, trace, U.g((), C.n()))
        assert isinstance(new_trace.get_sample(), Chm)
        assert isinstance(bwd_problem, Chm)


class TestFlag:
    def test_basic_operation(self):
        true_flags = [
            Flag(True),
            Flag(jnp.array(True)),
            Flag(jnp.array([True, True])),
            Flag(jnp.array([3.0, 4.0])),
        ]
        false_flags = [
            Flag(False),
            Flag(jnp.array(False)),
            Flag(jnp.array([True, False])),
            Flag(jnp.array([False, False])),
            Flag(jnp.array([0.0, 0.0])),
            Flag(jnp.array([0.0, 1.0])),
        ]
        for t in true_flags:
            assert t
            assert not t.not_()
            for f in false_flags:
                assert not f
                assert not t.and_(f)
                assert t.or_(f)
            for u in true_flags:
                assert t.and_(u)
                assert t.or_(u)

    def test_where(self):
        assert Flag(True).where(3.0, 4.0) == 3
        assert Flag(False).where(3.0, 4.0) == 4
        assert Flag(jnp.array(True)).where(3.0, 4.0) == 3
        assert Flag(jnp.array(False)).where(3.0, 4.0) == 4

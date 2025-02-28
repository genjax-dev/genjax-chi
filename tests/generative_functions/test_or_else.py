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

import jax
import pytest

import genjax


class TestOrElse:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_assess_or_else(self, key):
        @genjax.gen
        def f():
            return genjax.normal(0.0, 1.0) @ "value"

        f_or_f = f.or_else(f)
        args = (True, (), ())
        tr = f_or_f.simulate(key, args)
        score, ret = f_or_f.assess(f_or_f.simulate(key, args).get_choices(), args)

        assert tr.get_score() == score
        assert tr.get_retval() == ret

    def test_assess_or_else_inside_fn(self, key):
        p = 0.5

        @genjax.gen
        def f():
            flip = genjax.flip(p) @ "flip"
            return (
                genjax.normal(0.0, 1.0).or_else(genjax.normal(2.0, 1.0))(flip, (), ())
                @ "value"
            )

        args = ()
        tr = f.simulate(key, args)
        score, ret = f.assess(tr.get_choices(), args)

        assert tr.get_score() == score
        assert tr.get_retval() == ret

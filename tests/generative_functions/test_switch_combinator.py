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
import pytest
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax import numpy as jnp


class TestSwitchCombinator:
    def test_switch_combinator_simulate_in_gen_fn(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.gen
        def model():
            b = genjax.flip(0.5) @ "b"
            s = f.switch(f)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        assert 0.5672885 == tr.get_retval()

    def test_switch_combinator_simulate(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (0, (), ()))
        v1 = tr.get_sample()["y1"]
        v2 = tr.get_sample()["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        v1_score, _ = genjax.normal.assess(C.v(v1), (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        v2_score, _ = genjax.normal.assess(C.v(v2), (0.0, 1.0))
        assert score == v1_score + v2_score
        assert tr.get_args() == (0, (), ())
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (1, (), ()))
        b = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        (idx, *_) = tr.get_args()
        assert idx == 1

    def test_switch_combinator_choice_map_behavior(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        tr = jitted(key, (0, (), ()))
        assert "y1" in tr.get_sample()
        assert "y2" in tr.get_sample()
        assert "y3" not in tr.get_sample()

    def test_switch_combinator_importance(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.PRNGKey(314159)
        chm = C.n()
        jitted = jax.jit(switch.importance)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (0, (), ()))
        v1 = tr.get_sample().get_submap("y1")
        v2 = tr.get_sample().get_submap("y2")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        v1_score, _ = genjax.normal.assess(v1, (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        v2_score, _ = genjax.normal.assess(v2, (0.0, 1.0))
        assert score == v1_score + v2_score
        assert w == 0.0
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1, (), ()))
        b = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == 0.0
        chm = C["y3"].set(1)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1, (), ()))
        b = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == score

    def test_switch_combinator_update_single_branch_no_change(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        switch = simple_normal.switch()
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(switch.simulate)(sub_key, (0, ()))
        v1 = tr.get_sample()["y1"]
        v2 = tr.get_sample()["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (tr, _, _, _) = jax.jit(switch.update)(
            sub_key,
            tr,
            U.g(
                (Diff.no_change(0), ()),
                C.n(),
            ),
        )
        assert score == tr.get_score()
        assert v1 == tr.get_sample()["y1"]
        assert v2 == tr.get_sample()["y2"]

    def test_switch_combinator_update_updates_score(self):
        regular_stddev = 1.0
        outlier_stddev = 10.0
        sample_value = 2.0

        @genjax.gen
        def regular():
            x = genjax.normal(0.0, regular_stddev) @ "x"
            return x

        @genjax.gen
        def outlier():
            x = genjax.normal(0.0, outlier_stddev) @ "x"
            return x

        key = jax.random.PRNGKey(314159)
        switch = regular.switch(outlier)
        key, importance_key = jax.random.split(key)

        (tr, wt) = switch.importance(
            importance_key, C["x"].set(sample_value), (0, (), ())
        )
        (idx, *_) = tr.get_args()
        assert idx == 0
        assert (
            tr.get_score()
            == genjax.normal.assess(C.v(sample_value), (0.0, regular_stddev))[0]
        )
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        (new_tr, new_wt, _, _) = switch.update(
            update_key,
            tr,
            U.g(
                (Diff.unknown_change(1), (), ()),
                C.n(),
            ),
        )
        (idx, *_) = new_tr.get_args()
        assert idx == 1
        assert new_tr.get_score() != tr.get_score()
        assert tr.get_score() + new_wt == pytest.approx(new_tr.get_score(), 1e-5)

    def test_switch_combinator_vectorized_access(self):
        @genjax.gen
        def f1():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.gen
        def f2():
            return genjax.normal(0.0, 2.0) @ "y"

        s = f1.switch(f2)

        keys = jax.random.split(jax.random.PRNGKey(17), 3)
        # Just select 0 in all branches for simplicity:
        tr = jax.vmap(s.simulate, in_axes=(0, None))(keys, (0, (), ()))
        y = tr.get_choices()["y"]
        y = y.unmask()
        assert y.shape == (3,)

    def test_switch_combinator_with_empty_gen_fn(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.gen
        def empty():
            return 0.0

        @genjax.gen
        def model():
            b = genjax.flip(0.5) @ "b"
            s = f.switch(empty)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        assert 0.0 == tr.get_retval()

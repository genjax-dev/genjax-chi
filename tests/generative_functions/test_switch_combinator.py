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

import re

import jax
import pytest
from jax import numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax._src.core.typing import Array


class TestSwitch:
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

        tr = model.simulate(())
        assert tr.get_retval() == tr.get_choices()["s", "x"].unmask()

    def test_switch_combinator_simulate(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        jitted = jax.jit(switch.simulate)
        tr = jitted((0, (), ()))
        v1 = tr.get_choices()["y1"]
        v2 = tr.get_choices()["y2"]
        score = tr.get_score()
        v1_score, _ = genjax.normal.assess(C.v(v1), (0.0, 1.0))
        v2_score, _ = genjax.normal.assess(C.v(v2), (0.0, 1.0))
        assert score == v1_score + v2_score
        assert tr.get_args() == (0, (), ())
        tr = jitted((1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
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

        jitted = jax.jit(switch.simulate)
        tr = jitted((0, (), ()))
        chm = tr.get_choices()
        assert "y1" in chm
        assert "y2" in chm
        assert "y3" in chm
        assert chm["y3"] == genjax.Mask(jnp.array(False), jnp.array(False))

    def test_switch_combinator_importance(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        chm = C.n()
        jitted = jax.jit(switch.importance)
        (tr, w) = jitted(chm, (0, (), ()))
        v1 = tr.get_choices().get_submap("y1")
        v2 = tr.get_choices().get_submap("y2")
        score = tr.get_score()
        v1_score, _ = genjax.normal.assess(v1, (0.0, 1.0))
        v2_score, _ = genjax.normal.assess(v2, (0.0, 1.0))
        assert score == v1_score + v2_score
        assert w == 0.0
        (tr, w) = jitted(chm, (1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == 0.0
        chm = C["y3"].set(1)
        (tr, w) = jitted(chm, (1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == score

    def test_switch_combinator_update_single_branch_no_change(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        switch = simple_normal.switch()
        tr = jax.jit(switch.simulate)((0, ()))
        v1 = tr.get_choices()["y1"]
        v2 = tr.get_choices()["y2"]
        score = tr.get_score()
        (tr, _, _, _) = jax.jit(switch.update)(
            tr,
            C.n(),
            (Diff.no_change(0), ()),
        )
        assert score == tr.get_score()
        assert v1 == tr.get_choices()["y1"]
        assert v2 == tr.get_choices()["y2"]

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

        switch = regular.switch(outlier)

        (tr, wt) = switch.importance(C["x"].set(sample_value), (0, (), ()))
        (idx, *_) = tr.get_args()
        assert idx == 0
        assert (
            tr.get_score()
            == genjax.normal.assess(C.v(sample_value), (0.0, regular_stddev))[0]
        )
        assert wt == tr.get_score()

        (new_tr, new_wt, _, _) = switch.update(
            tr,
            C.n(),
            (Diff.unknown_change(1), (), ()),
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

        # Just select 0 in all branches for simplicity:
        tr = jax.vmap(lambda k, args: s.simulate(args), in_axes=(0, None))(
            jnp.zeros(3), (0, (), ())
        )
        y = tr.get_choices()["y"].unmask()
        assert y.shape == (3,)

    def test_switch_combinator_with_empty_gen_fn(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.gen
        def empty():
            return jnp.asarray(0.0)

        @genjax.gen
        def model():
            b = genjax.flip(0.5) @ "b"
            s = f.switch(empty)(jnp.int32(b), (), ()) @ "s"
            return s

        chm = C["b"].set(1)
        tr, _ = model.importance(chm, ())
        assert 0.0 == tr.get_retval()

    def test_switch_combinator_with_different_return_types(self):
        @genjax.gen
        def identity(x: int) -> Array:
            return jnp.asarray(x)

        @genjax.gen
        def bool_branch(_: int) -> Array:
            return jnp.asarray(True)

        switch_model = genjax.switch(identity, bool_branch)

        bare_idx_result = switch_model(1, (10,), (10,))()
        assert bare_idx_result == jnp.asarray(1)
        assert bare_idx_result.dtype == jnp.int32

        # this case returns 1
        array_idx_result = switch_model(jnp.array(1), (10,), (10,))()
        assert array_idx_result == jnp.asarray(1)
        assert array_idx_result.dtype == bare_idx_result.dtype

    def test_runtime_incompatible_types(self):
        @genjax.gen
        def three_branch(x: int):
            return jax.numpy.ones(3)

        @genjax.gen
        def four_branch(_: int):
            return jax.numpy.ones(4)

        switch_model = three_branch.switch(four_branch)

        with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
            switch_model(0, (10,), (10,))()

    def test_switch_distinct_addresses(self):
        @genjax.gen
        def x_z():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal(x, jnp.ones(3)) @ "z"
            return x

        @genjax.gen
        def x_y():
            x = genjax.normal(0.0, 2.0) @ "x"
            _ = genjax.normal(x, jnp.ones(20)) @ "y"
            return x

        model = x_z.switch(x_y)
        tr = model.simulate((jnp.array(0), (), ()))

        # both xs match, so it's fine to combine across models
        assert tr.get_choices()["x"].unmask().shape == ()

        # y and z only show up on one side of the `switch` so any shape is fine
        assert tr.get_choices()["y"].unmask().shape == (20,)
        assert tr.get_choices()["z"].unmask().shape == (3,)

        @genjax.gen
        def arr_x():
            _ = genjax.normal(0.0, jnp.array([2.0, 2.0])) @ "x"
            _ = genjax.normal(0.0, jnp.ones(20)) @ "y"
            return jnp.array(1.0)

        mismatched_tr = x_z.switch(arr_x).simulate((jnp.array(0), (), ()))

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot combine masks with different array shapes: () vs (2,)"
            ),
        ):
            mismatched_tr.get_choices()["x"]

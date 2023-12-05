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
from jax import numpy as jnp

import genjax
from genjax import Mask
from genjax.incremental import NoChange
from genjax.incremental import UnknownChange
from genjax.incremental import diff
from genjax.incremental import tree_diff_no_change


class TestSwitch:
    def test_switch_simulate_in_gen_fn(self):
        @genjax.lang(genjax.Static)
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.lang(genjax.Static)
        def model():
            b = genjax.bernoulli(0.5) @ "b"
            s = genjax.Switch(f, f)(jnp.int32(b)) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        assert True

    def test_switch_simulate(self):
        @genjax.lang(genjax.Static)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.lang(genjax.Static)
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.Switch(simple_normal, simple_bernoulli)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (0,))
        v1 = tr.get_submap("y1")
        v2 = tr.get_submap("y2")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, v1_score) = genjax.normal.importance(sub_key, v1, (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        (_, v2_score) = genjax.normal.importance(sub_key, v2, (0.0, 1.0))
        assert score == v1_score + v2_score
        assert tr.get_args() == (0,)
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (1,))
        flip = tr.get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(sub_key, flip, (0.3,))
        assert score == flip_score
        assert tr.get_args() == (1,)

    def test_switch_simulate_in_gen_fn(self):
        @genjax.lang(genjax.Static)
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.lang(genjax.Static)
        def model():
            b = genjax.bernoulli(0.5) @ "b"
            s = genjax.Switch(f, f)(jnp.int32(b)) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        tr = model.simulate(key, ())
        assert True

    def test_switch_choice_map_behavior(self):
        @genjax.lang(genjax.Static)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.lang(genjax.Static)
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.Switch(simple_normal, simple_bernoulli)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        tr = jitted(key, (0,))
        assert isinstance(tr["y1"], Mask)
        assert isinstance(tr["y2"], Mask)
        assert isinstance(tr["y3"], Mask)

    def test_switch_importance(self):
        @genjax.lang(genjax.Static)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.lang(genjax.Static)
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.Switch(simple_normal, simple_bernoulli)

        key = jax.random.PRNGKey(314159)
        chm = genjax.EmptyChoice()
        jitted = jax.jit(switch.importance)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (0,))
        v1 = tr.get_submap("y1")
        v2 = tr.get_submap("y2")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, v1_score) = genjax.normal.importance(
            sub_key,
            v1,
            (0.0, 1.0),
        )
        key, sub_key = jax.random.split(key)
        (_, v2_score) = genjax.normal.importance(
            sub_key,
            v2,
            (0.0, 1.0),
        )
        assert score == v1_score + v2_score
        assert w == 0.0
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1,))
        flip = tr.get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(
            sub_key,
            flip,
            (0.3,),
        )
        assert score == flip_score
        assert w == 0.0
        chm = genjax.choice_map({"y3": 1})
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1,))
        flip = tr.get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(
            sub_key,
            flip,
            (0.3,),
        )
        assert score == flip_score
        assert w == score

    def test_switch_update_single_branch_no_change(self):
        @genjax.lang(genjax.Static)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        switch = genjax.Switch(simple_normal)
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(switch.simulate)(sub_key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (tr, _, _, _) = jax.jit(switch.update)(
            sub_key, tr, genjax.empty_choice(), (diff(0, NoChange),)
        )
        assert score == tr.get_score()
        assert v1 == tr["y1"]
        assert v2 == tr["y2"]

    def test_switch_update_with_masking(self):
        @genjax.lang(genjax.Static)
        def branch_1(v):
            return genjax.normal(v, 1.0) @ "v"

        @genjax.lang(genjax.Static)
        def branch_2(v):
            return genjax.normal(v, 3.0) @ "v"

        switch = genjax.Switch(branch_1, branch_2)
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(switch.simulate)(key, (1, 0.0))
        (tr, w, rd, d) = jax.jit(switch.update)(
            key,
            tr,
            genjax.mask(jnp.array(True), genjax.empty_choice()),
            tree_diff_no_change((1, 0.0)),
        )
        assert isinstance(d, genjax.EmptyChoice)
        assert w == 0.0
        (tr, w, rd, d) = jax.jit(switch.update)(
            key,
            tr,
            genjax.mask(jnp.array(False), genjax.empty_choice()),
            tree_diff_no_change((1, 0.0)),
        )
        assert isinstance(d, genjax.EmptyChoice)
        assert w == 0.0

    def test_switch_update_updates_score(self):
        regular_stddev = 1.0
        outlier_stddev = 10.0
        sample_value = 2.0

        @genjax.lang(genjax.Static)
        def regular():
            x = genjax.normal(0.0, regular_stddev) @ "x"
            return x

        @genjax.lang(genjax.Static)
        def outlier():
            x = genjax.normal(0.0, outlier_stddev) @ "x"
            return x

        key = jax.random.PRNGKey(314159)
        switch = genjax.Switch(regular, outlier)
        key, importance_key = jax.random.split(key)

        (tr, wt) = switch.importance(
            importance_key, genjax.choice_map({"x": sample_value}), (0,)
        )
        assert tr.chm.index == 0
        assert tr.get_score() == genjax.normal.logpdf(sample_value, 0.0, regular_stddev)
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        (new_tr, new_wt, _, _) = switch.update(
            update_key,
            tr,
            genjax.empty_choice(),
            (diff(1, UnknownChange),),
        )
        assert new_tr.chm.index == 1
        assert new_tr.get_score() != tr.get_score()
        assert new_tr.get_score() == genjax.normal.logpdf(
            sample_value, 0.0, outlier_stddev
        )

    def test_switch_vectorized_access(self):
        @genjax.lang(genjax.Static)
        def f1():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.lang(genjax.Static)
        def f2():
            return genjax.normal(0.0, 2.0) @ "y"

        s = genjax.Switch(f1, f2)

        keys = jax.random.split(jax.random.PRNGKey(17), 3)
        # Just select 0 in all branches for simplicity:
        tr = jax.vmap(s.simulate, in_axes=(0, None))(keys, (0,))
        y = tr["y"]
        v = y.unsafe_unmask()
        assert v.shape == (3,)
        assert (y.mask == jnp.array([True, True, True])).all()

    def test_switch_with_empty_gen_fn(self):
        @genjax.lang(genjax.Static)
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.lang(genjax.Static)
        def empty():
            return 0.0

        @genjax.lang(genjax.Static)
        def model():
            b = genjax.bernoulli(0.5) @ "b"
            s = genjax.Switch(f, empty)(jnp.int32(b)) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        assert True

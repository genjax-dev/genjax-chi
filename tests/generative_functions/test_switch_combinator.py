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
from genjax import ChoiceMap as C
from genjax import Mask
from genjax.incremental import Diff, NoChange, UnknownChange
from jax import numpy as jnp


class TestSwitchCombinator:
    def test_switch_combinator_simulate_in_gen_fn(self):
        @genjax.static_gen_fn
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.static_gen_fn
        def model():
            b = genjax.bernoulli(0.5) @ "b"
            s = genjax.switch_combinator(f, f)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        _tr = model.simulate(sub_key, ())
        assert True

    def test_switch_combinator_simulate(self):
        @genjax.static_gen_fn
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.static_gen_fn
        def simple_bernoulli():
            _y3 = genjax.bernoulli(0.3) @ "y3"

        switch = genjax.switch_combinator(simple_normal, simple_bernoulli)

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
        tr = jitted(sub_key, (1,))
        flip = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(sub_key, flip, (0.3,))
        assert score == flip_score
        assert tr.get_args() == (1,)

    def test_switch_combinator_choice_map_behavior(self):
        @genjax.static_gen_fn
        def simple_normal():
            _y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            _y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.static_gen_fn
        def simple_bernoulli():
            _y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.switch_combinator(simple_normal, simple_bernoulli)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        tr = jitted(key, (0,))
        assert isinstance(tr.get_sample().get_submap("y1"), Mask)
        assert isinstance(tr.get_sample().get_submap("y2"), Mask)
        assert isinstance(tr.get_sample().get_submap("y3"), Mask)

    def test_switch_combinator_importance(self):
        @genjax.static_gen_fn
        def simple_normal():
            _y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            _y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.static_gen_fn
        def simple_bernoulli():
            _y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.switch_combinator(simple_normal, simple_bernoulli)

        key = jax.random.PRNGKey(314159)
        chm = C.n
        jitted = jax.jit(switch.importance)
        key, sub_key = jax.random.split(key)
        (tr, w, _) = jitted(sub_key, chm, (0, (), ()))
        v1 = tr.get_sample().get_submap("y1")
        v2 = tr.get_sample().get_submap("y2")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        v1_score, _ = genjax.normal.assess(C.v(v1), (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        v2_score, _ = genjax.normal.assess(C.v(v2), (0.0, 1.0))
        assert score == v1_score + v2_score
        assert w == 0.0
        key, sub_key = jax.random.split(key)
        (tr, w, _) = jitted(sub_key, chm, (1, (), ()))
        flip = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(
            sub_key,
            flip,
            (0.3,),
        )
        assert score == flip_score
        assert w == 0.0
        chm = genjax.C.n.at["y3"].set(1)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1,))
        flip = tr.get_sample().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (_, flip_score) = genjax.bernoulli.importance(
            sub_key,
            flip,
            (0.3,),
        )
        assert score == flip_score
        assert w == score

    def test_switch_combinator_update_single_branch_no_change(self):
        @genjax.static_gen_fn
        def simple_normal():
            _y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            _y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        switch = genjax.switch_combinator(simple_normal)
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(switch.simulate)(sub_key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (tr, _, _, _) = jax.jit(switch.update)(
            sub_key, tr, genjax.EmptyChoice(), (Diff.tree_diff(0, NoChange),)
        )
        assert score == tr.get_score()
        assert v1 == tr["y1"]
        assert v2 == tr["y2"]

    def test_switch_combinator_update_with_masking(self):
        @genjax.static_gen_fn
        def branch_1(v):
            return genjax.normal(v, 1.0) @ "v"

        @genjax.static_gen_fn
        def branch_2(v):
            return genjax.normal(v, 3.0) @ "v"

        switch = genjax.switch_combinator(branch_1, branch_2)
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(switch.simulate)(key, (1, 0.0))
        (tr, w, rd, d) = jax.jit(switch.update)(
            key,
            tr,
            Mask(jnp.array(True), genjax.EmptyChoice()),
            Diff.tree_diff_no_change((1, 0.0)),
        )
        assert isinstance(d, genjax.EmptyChoice)
        assert w == 0.0
        (tr, w, rd, d) = jax.jit(switch.update)(
            key,
            tr,
            Mask(jnp.array(False), genjax.EmptyChoice()),
            Diff.tree_diff_no_change((1, 0.0)),
        )
        assert isinstance(d, genjax.EmptyChoice)
        assert w == 0.0

    def test_switch_combinator_update_updates_score(self):
        regular_stddev = 1.0
        outlier_stddev = 10.0
        sample_value = 2.0

        @genjax.static_gen_fn
        def regular():
            x = genjax.normal(0.0, regular_stddev) @ "x"
            return x

        @genjax.static_gen_fn
        def outlier():
            x = genjax.normal(0.0, outlier_stddev) @ "x"
            return x

        key = jax.random.PRNGKey(314159)
        switch = genjax.switch_combinator(regular, outlier)
        key, importance_key = jax.random.split(key)

        (tr, wt) = switch.importance(
            importance_key, genjax.choice_map({"x": sample_value}), (0,)
        )
        assert tr.choice.index == 0
        assert tr.get_score() == genjax.normal.logpdf(sample_value, 0.0, regular_stddev)
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        (new_tr, new_wt, _, _) = switch.update(
            update_key,
            tr,
            genjax.EmptyChoice(),
            (Diff.tree_diff(1, UnknownChange),),
        )
        assert new_tr.choice.index == 1
        assert new_tr.get_score() != tr.get_score()
        assert new_tr.get_score() == genjax.normal.logpdf(
            sample_value, 0.0, outlier_stddev
        )

    def test_switch_combinator_vectorized_access(self):
        @genjax.static_gen_fn
        def f1():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.static_gen_fn
        def f2():
            return genjax.normal(0.0, 2.0) @ "y"

        s = genjax.switch_combinator(f1, f2)

        keys = jax.random.split(jax.random.PRNGKey(17), 3)
        # Just select 0 in all branches for simplicity:
        tr = jax.vmap(s.simulate, in_axes=(0, None))(keys, (0, (), ()))
        y = tr.get_sample()["y"]
        assert y.shape == (3,)

    def test_switch_combinator_with_empty_gen_fn(self):
        @genjax.static_gen_fn
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.static_gen_fn
        def empty():
            return 0.0

        @genjax.static_gen_fn
        def model():
            b = genjax.bernoulli(0.5) @ "b"
            s = genjax.switch_combinator(f, empty)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        _tr = model.simulate(sub_key, ())
        assert True

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
from jax import numpy as jnp

import genjax
from genjax import BooleanMask
from genjax import NoChange
from genjax import UnknownChange
from genjax import diff


class TestSimulate:
    def test_switch_simulate_in_gen_fn(self):
        @genjax.gen
        def f():
            x = genjax.tfp_normal(0., 1.) @ 'x'
            return x

        @genjax.gen
        def model():
            b = genjax.bernoulli(.5) @ 'b'
            s = genjax.Switch(f, f)(jnp.int32(b)) @ 's'  
            return s

        key = jax.random.PRNGKey(314159)
        # This crashes.
        _, tr = genjax.simulate(model)(key, ())
        
    def test_switch_simulate(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.gen
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.SwitchCombinator([simple_normal, simple_bernoulli])

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        key, tr = jitted(key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, (v1_score, _) = genjax.normal.importance(
            key, genjax.value_choice_map(v1), (0.0, 1.0)
        )
        key, (v2_score, _) = genjax.normal.importance(
            key, genjax.value_choice_map(v2), (0.0, 1.0)
        )
        assert score == v1_score + v2_score
        assert tr.get_args() == (0,)
        key, tr = jitted(key, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        key, (flip_score, _) = genjax.bernoulli.importance(
            key, genjax.value_choice_map(flip), (0.3,)
        )
        assert score == flip_score
        assert tr.get_args() == (1,)

    def test_switch_choice_map_behavior(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.gen
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.SwitchCombinator([simple_normal, simple_bernoulli])

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        key, tr = jitted(key, (0,))
        assert isinstance(tr["y1"], BooleanMask)
        assert isinstance(tr["y2"], BooleanMask)
        assert isinstance(tr["y3"], BooleanMask)

    def test_switch_importance(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        @genjax.gen
        def simple_bernoulli():
            y3 = genjax.trace("y3", genjax.bernoulli)(0.3)

        switch = genjax.SwitchCombinator([simple_normal, simple_bernoulli])

        key = jax.random.PRNGKey(314159)
        chm = genjax.EmptyChoiceMap()
        jitted = jax.jit(switch.importance)
        key, (w, tr) = jitted(key, chm, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, (v1_score, _) = genjax.normal.importance(
            key,
            genjax.value_choice_map(v1),
            (0.0, 1.0),
        )
        key, (v2_score, _) = genjax.normal.importance(
            key,
            genjax.value_choice_map(v2),
            (0.0, 1.0),
        )
        assert score == v1_score + v2_score
        assert w == 0.0
        key, (w, tr) = jitted(key, chm, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        key, (flip_score, _) = genjax.bernoulli.importance(
            key, genjax.value_choice_map(flip), (0.3,)
        )
        assert score == flip_score
        assert w == 0.0
        chm = genjax.choice_map({"y3": True})
        key, (w, tr) = jitted(key, chm, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        key, (flip_score, _) = genjax.bernoulli.importance(
            key, genjax.value_choice_map(flip), (0.3,)
        )
        assert score == flip_score
        assert w == score

    def test_switch_update_single_branch_no_change(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)

        switch = genjax.SwitchCombinator([simple_normal])
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(switch.simulate)(key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, (_, _, tr, _) = jax.jit(switch.update)(
            key, tr, genjax.empty_choice_map(), (diff(0, NoChange),)
        )
        assert score == tr.get_score()
        assert v1 == tr["y1"]
        assert v2 == tr["y2"]

    def test_switch_update_updates_score(self):
        key = jax.random.PRNGKey(314159)

        regular_stddev = 1.0
        outlier_stddev = 10.0
        sample_value = 2.0

        @genjax.gen
        def regular():
            x = genjax.tfp_normal(0.0, regular_stddev) @ "x"
            return x

        @genjax.gen
        def outlier():
            x = genjax.tfp_normal(0.0, outlier_stddev) @ "x"
            return x

        switch = genjax.SwitchCombinator([regular, outlier])
        key, importance_key = jax.random.split(key)

        _, (wt, tr) = switch.importance(
            importance_key, genjax.choice_map({"x": sample_value}), (0,)
        )
        assert tr.chm.index == 0
        assert tr.get_score() == genjax.tfp_normal.logpdf(
            sample_value, 0.0, regular_stddev
        )
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        _, (_, new_wt, new_tr, _) = switch.update(
            update_key,
            tr,
            genjax.empty_choice_map(),
            (diff(1, UnknownChange),),
        )
        assert new_tr.chm.index == 1
        assert new_tr.get_score() != tr.get_score()
        assert new_tr.get_score() == genjax.tfp_normal.logpdf(
            sample_value, 0.0, outlier_stddev
        )

    def test_switch_vectorized_access(self):
        @genjax.gen
        def f1():
            return genjax.tfp_normal(0.0, 1.0) @ "y"

        @genjax.gen
        def f2():
            return genjax.tfp_normal(0.0, 2.0) @ "y"

        s = genjax.SwitchCombinator([f1, f2])

        keys = jax.random.split(jax.random.PRNGKey(17), 3)
        # Just select 0 in all branches for simplicity:
        _, tr = jax.vmap(s.simulate, in_axes=(0, None))(keys, (0,))
        y = tr["y"]
        assert y.value.shape == (3,)
        assert (y.mask == jnp.array([True, True, True])).all()

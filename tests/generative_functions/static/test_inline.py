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
import pytest

import genjax


class TestInline:
    def test_inline_simulate(self):
        @genjax.gen(genjax.Static)
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.gen(genjax.Static)
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.gen(genjax.Static)
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        choices = tr.strip()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")
        tr = jax.jit(higher_higher_model.simulate)(key, ())
        choices = tr.strip()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")

    def test_inline_importance(self):
        @genjax.gen(genjax.Static)
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.gen(genjax.Static)
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.gen(genjax.Static)
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({"y1": 3.0})
        key, sub_key = jax.random.split(key)
        (tr, w) = jax.jit(higher_model.importance)(sub_key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)
        (tr, w) = jax.jit(higher_higher_model.importance)(key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)

    def test_inline_update(self):
        @genjax.gen(genjax.Static)
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.gen(genjax.Static)
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.gen(genjax.Static)
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        chm = genjax.choice_map({"y1": 3.0})
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        old_value = tr.strip()["y1"]
        key, sub_key = jax.random.split(key)
        (tr, w, rd, _) = jax.jit(higher_model.update)(sub_key, tr, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(
            choices["y1"], 0.0, 1.0
        ) - genjax.normal.logpdf(old_value, 0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_higher_model.simulate)(sub_key, ())
        old_value = tr.strip()["y1"]
        (tr, w, rd, _) = jax.jit(higher_higher_model.update)(key, tr, chm, ())
        choices = tr.strip()
        assert w == pytest.approx(
            genjax.normal.logpdf(choices["y1"], 0.0, 1.0)
            - genjax.normal.logpdf(old_value, 0.0, 1.0),
            0.0001,
        )

    def test_inline_assess(self):
        @genjax.gen(genjax.Static)
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.gen(genjax.Static)
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.gen(genjax.Static)
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({"y1": 3.0, "y2": 3.0})
        (score, ret) = jax.jit(higher_model.assess)(chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)
        (score, ret) = jax.jit(higher_higher_model.assess)(chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)

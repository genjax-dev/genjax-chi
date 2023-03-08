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

import genjax


@genjax.gen
def simple_normal():
    y1 = genjax.normal(0.0, 1.0) @ "y1"
    y2 = genjax.normal(0.0, 1.0) @ "y2"
    return y1 + y2


@genjax.gen
def higher_model():
    y = simple_normal.inline()
    return y


@genjax.gen
def higher_higher_model():
    y = higher_model.inline()
    return y


class TestInline:
    def test_inline_simulate(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(higher_model.simulate)(key, ())
        choices = tr.strip()
        assert choices.has_subtree("y1")
        assert choices.has_subtree("y2")
        key, tr = jax.jit(higher_higher_model.simulate)(key, ())
        choices = tr.strip()
        assert choices.has_subtree("y1")
        assert choices.has_subtree("y2")

    def test_inline_importance(self):
        key = jax.random.PRNGKey(314159)
        chm = genjax.chm({"y1": 3.0})
        key, (w, tr) = jax.jit(higher_model.importance)(key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)
        key, (w, tr) = jax.jit(higher_higher_model.importance)(key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)

    def test_inline_update(self):
        key = jax.random.PRNGKey(314159)
        chm = genjax.chm({"y1": 3.0})
        key, tr = jax.jit(higher_model.simulate)(key, ())
        old_value = tr.strip()["y1"]
        key, (rd, w, tr, _) = jax.jit(higher_model.update)(key, tr, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(
            choices["y1"], 0.0, 1.0
        ) - genjax.normal.logpdf(old_value, 0.0, 1.0)
        key, tr = jax.jit(higher_higher_model.simulate)(key, ())
        old_value = tr.strip()["y1"]
        key, (rd, w, tr, _) = jax.jit(higher_higher_model.update)(key, tr, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(
            choices["y1"], 0.0, 1.0
        ) - genjax.normal.logpdf(old_value, 0.0, 1.0)

    def test_inline_assess(self):
        key = jax.random.PRNGKey(314159)
        chm = genjax.chm({"y1": 3.0, "y2": 3.0})
        key, (ret, score) = jax.jit(higher_model.assess)(key, chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)
        key, (ret, score) = jax.jit(higher_higher_model.assess)(key, chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)

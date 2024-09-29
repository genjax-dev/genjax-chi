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

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMap as C
from genjax import ChoiceMapConstraint, Diff, Index, Regenerate, Update
from genjax import SelectionBuilder as S


class TestRegenerate:
    def test_simple_normal_update(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        model_tr = simple_normal.simulate(sub_key, ())
        request = Update(ChoiceMapConstraint(C.kw(y1=3.0)))
        old_value = model_tr.get_choices()["y1"]
        new_tr, fwd_w, _, bwd_request = request.edit(key, model_tr, ())
        new_value = new_tr.get_choices()["y1"]
        assert fwd_w != 0.0
        assert new_value != old_value
        old_tr, bwd_w, _, _ = bwd_request.edit(
            key,
            new_tr,
            (),
        )
        old_old_value = old_tr.get_choices()["y1"]
        assert old_old_value == old_value
        assert bwd_w != 0.0
        assert fwd_w + bwd_w == 0.0
        assert model_tr.get_score() == old_tr.get_score()

    def test_simple_normal_regenerate(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        # First, try y1
        old_v = tr.get_choices()["y1"]
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        assert fwd_w != 0.0
        new_v = new_tr.get_choices()["y1"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(key, new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y1"]
        assert old_old_v == old_v

        # Now, do y2
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(S["y2"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        assert fwd_w != 0.0
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(key, new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v

        # What about both?
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(
            S["y1"] | S["y2"],
        )
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, _ = bwd_request.edit(key, new_tr, ())
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v
        assert tr.get_score() == old_tr.get_score()


class TestIndex:
    def test_simple_scan_index_update(self):
        @genjax.gen
        def kernel(carry, scanned_in):
            y1 = genjax.normal(carry, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2, None

        model = kernel.scan(n=10)
        key = jax.random.PRNGKey(314159)
        args = (0.0, None)
        model_tr = model.simulate(key, args)
        request = Index(jnp.array(3), Update(ChoiceMapConstraint(C.kw(y1=3.0))))
        old_value = model_tr.get_choices()[3, "y1"]
        new_tr, fwd_w, _, bwd_request = request.edit(
            key,
            model_tr,
            Diff.no_change(args),
        )
        new_value = new_tr.get_choices()[3, "y1"]
        assert fwd_w != 0.0
        assert new_value != old_value
        old_tr, bwd_w, _, _ = bwd_request.edit(
            key,
            new_tr,
            Diff.no_change(args),
        )
        old_old_value = old_tr.get_choices()[3, "y1"]
        assert old_old_value == old_value
        assert bwd_w != 0.0
        assert fwd_w + bwd_w == 0.0
        assert model_tr.get_score() == pytest.approx(old_tr.get_score(), 1e-4)

    def test_simple_scan_index_regenerate(self):
        @genjax.gen
        def kernel(carry, scanned_in):
            y1 = genjax.normal(carry, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2, None

        model = kernel.scan(n=10)
        key = jax.random.PRNGKey(314159)
        args = (0.0, None)
        model_tr = model.simulate(key, args)
        request = Index(jnp.array(3), Regenerate(S["y1"]))
        old_value = model_tr.get_choices()[3, "y1"]
        new_tr, fwd_w, _, bwd_request = request.edit(
            key,
            model_tr,
            Diff.no_change(args),
        )
        new_value = new_tr.get_choices()[3, "y1"]
        assert fwd_w != 0.0
        assert new_value != old_value
        old_tr, bwd_w, _, _ = bwd_request.edit(
            key,
            new_tr,
            Diff.no_change(args),
        )
        old_old_value = old_tr.get_choices()[3, "y1"]
        assert old_old_value == old_value
        assert bwd_w != 0.0
        assert fwd_w + bwd_w == 0.0
        assert model_tr.get_score() == old_tr.get_score()

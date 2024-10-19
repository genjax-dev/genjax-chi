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
import jax.random as jrand
import pytest

import genjax
from genjax import ChoiceMap, Diff, DiffAnnotate, EmptyRequest, Regenerate, Selection
from genjax import SelectionBuilder as S
from genjax._src.generative_functions.static import StaticRequest
from genjax.inference.requests import HMC, SafeHMC


class TestRegenerate:
    def test_simple_normal_regenerate(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        # First, try y1
        old_v = tr.get_choices()["y1"]
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        assert fwd_w != 0.0
        new_v = new_tr.get_choices()["y1"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(sub_key, new_tr, ())
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
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(key, new_tr, ())
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v


class TestHMC:
    def test_simple_normal_hmc(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 0.01) @ "y"
            return y

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        tr, _ = model.importance(sub_key, ChoiceMap.kw(y=3.0), ())
        request = HMC(Selection.at["x"], jnp.array(1e-2))
        editor = jax.jit(request.edit)
        new_tr = tr
        for _ in range(20):
            key, sub_key = jrand.split(key)
            new_tr, *_ = editor(sub_key, new_tr, ())
        assert new_tr.get_choices()["x"] == pytest.approx(3.0, 5e-3)

    def test_simple_scan_hmc(self):
        @genjax.gen
        def kernel(z, scanned_in):
            z = genjax.normal(z, 1.0) @ "x"
            _ = genjax.normal(z, 0.01) @ "y"
            return z, None

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        model = kernel.scan(n=10)
        vchm = jax.vmap(lambda idx: ChoiceMap.empty().at[idx, "y"].set(3.0))(
            jnp.arange(10)
        )
        tr, _ = model.importance(sub_key, vchm, (0.0, None))
        request = HMC(Selection.at[..., "x"], jnp.array(1e-2))
        editor = jax.jit(request.edit)
        new_tr = tr
        for _ in range(20):
            key, sub_key = jrand.split(key)
            new_tr, *_ = editor(sub_key, new_tr, Diff.no_change((0.0, None)))
        assert new_tr.get_choices()[..., "x"] == pytest.approx(3.0, 8e-3)

    def test_safe_hmc(self):
        @genjax.gen
        def submodel():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 0.01) @ "y"
            return y

        @genjax.gen
        def model():
            _ = submodel() @ "x"
            _ = submodel() @ "y"

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        tr, _ = model.importance(sub_key, ChoiceMap.kw(y=3.0), ())
        request = StaticRequest(
            {"x": SafeHMC(Selection.at["x"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        new_tr, w, *_ = editor(sub_key, tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert w != 0.0

        # Compositional request with HMC.
        request = StaticRequest(
            {
                "x": SafeHMC(Selection.at["x"], jnp.array(1e-2)),
                "y": Regenerate(Selection.at["x"]),
            },
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        new_tr, w, *_ = editor(sub_key, tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert new_tr.get_choices()["y", "x"] != tr.get_choices()["y", "x"]
        assert w != 0.0

        request = StaticRequest(
            {"x": SafeHMC(Selection.at["y"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        with pytest.raises(Exception):
            new_tr, w, *_ = editor(sub_key, tr, ())


class TestDiffCoercion:
    def test_diff_coercion(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        # Test that DiffCoercion.edit is being
        # properly used compositionally.
        def assert_no_change(v):
            assert Diff.static_check_no_change(v)
            return v

        request = StaticRequest({
            "y1": Regenerate(Selection.all()),
            "y2": DiffAnnotate(
                EmptyRequest(),
                argdiff_fn=assert_no_change,
            ),
        })

        with pytest.raises(Exception):
            request.edit(key, tr, ())

        # Test equivalent between requests which use
        # DiffCoercion in trivial ways.
        unwrapped_request = StaticRequest({
            "y1": Regenerate(Selection.all()),
        })
        wrapped_request = StaticRequest({
            "y1": Regenerate(Selection.all()).contramap(assert_no_change),
            "y2": EmptyRequest().map(assert_no_change),
        })
        _, w, _, _ = unwrapped_request.edit(key, tr, ())
        _, w_, _, _ = wrapped_request.edit(key, tr, ())
        assert w == w_

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
import jax.tree_util as jtu
import pytest

import genjax
from genjax import (
    ChoiceMap,
    Diff,
    DiffAnnotate,
    EmptyRequest,
    Regenerate,
    Selection,
    Update,
)
from genjax import ChoiceMap as C
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

        # First, try y1 and test for correctness.
        old_v = tr.get_choices()["y1"]
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y1"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
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
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y2"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
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

    def test_linked_normal_regenerate(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(y1, 1.0) @ "y2"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = linked_normal.simulate(sub_key, ())

        # First, try y1 and test for correctness.
        old_y1 = tr.get_choices()["y1"]
        old_y2 = tr.get_choices()["y2"]
        old_target_density = genjax.normal.logpdf(
            old_y1, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y2, old_y1, 1.0)
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, _ = request.edit(key, tr, ())
        new_y1 = new_tr.get_choices()["y1"]
        new_y2 = new_tr.get_choices()["y2"]
        new_target_density = genjax.normal.logpdf(
            new_y1, 0.0, 1.0
        ) + genjax.normal.logpdf(new_y2, new_y1, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == pytest.approx(new_target_density - old_target_density, 1e-6)

    def test_linked_normal_convergence(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 3.0) @ "y1"
            _ = genjax.normal(y1, 0.01) @ "y2"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr, _ = linked_normal.importance(sub_key, C.kw(y2=3.0), ())
        request = Regenerate(S["y1"])

        # Run Metropolis-Hastings for 100 steps.
        for _ in range(100):
            key, sub_key = jax.random.split(key)
            new_tr, w, _, _ = request.edit(sub_key, tr, ())
            key, sub_key = jax.random.split(key)
            check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
            tr = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), new_tr, tr)

        assert tr.get_choices()["y1"] == pytest.approx(3.0, 1e-3)


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

        # First, try moving x and test for correctness.
        old_x = tr.get_choices()["x"]
        old_y = tr.get_choices()["y"]
        old_target_density = genjax.normal.logpdf(
            old_x, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y, old_x, 0.01)
        new_tr, fwd_w, _, _ = editor(key, tr, ())
        new_x = new_tr.get_choices()["x"]
        new_y = new_tr.get_choices()["y"]
        new_target_density = genjax.normal.logpdf(
            new_x, 0.0, 1.0
        ) + genjax.normal.logpdf(new_y, new_x, 0.01)
        assert fwd_w != 0.0
        # The change in the target scores corresponds to the non-momenta terms in the HMC alpha computation.
        assert (new_tr.get_score() - tr.get_score()) == pytest.approx(
            new_target_density - old_target_density, 1e-6
        )
        # The weight factors in the target score change and the momenta, so removing the change in the target scores should leave us with a non-zero contribution from the momenta.
        assert fwd_w - (new_tr.get_score() - tr.get_score()) != 0.0

        # Check for gradient convergence.
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

        # Compositional request _including_ HMC.
        request = StaticRequest(
            {
                "x": SafeHMC(Selection.at["x"], jnp.array(1e-2)),
                "y": StaticRequest({
                    "x": Regenerate(Selection.all()),
                    "y": Update(C.choice(3.0)),
                }),
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

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
from genjax._src.generative_functions.gen import DefRequest
from genjax.edits import HMC, Rejuvenate, SafeHMC


class TestRegenerate:
    def test_simple_normal_regenerate(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        tr = simple_normal.simulate(())

        # First, try y1 and test for correctness.
        old_v = tr.get_choices()["y1"]
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, bwd_request = request.edit(tr, ())
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y1"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
        new_v = new_tr.get_choices()["y1"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y1"]
        assert old_old_v == old_v

        # Now, do y2
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(S["y2"])
        new_tr, fwd_w, _, bwd_request = request.edit(tr, ())
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y2"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v

        # What about both?
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(
            S["y1"] | S["y2"],
        )
        new_tr, fwd_w, _, bwd_request = request.edit(tr, ())
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(new_tr, ())
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v

    def test_linked_normal_regenerate(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(y1, 1.0) @ "y2"

        tr = linked_normal.simulate(())

        # First, try y1 and test for correctness.
        old_y1 = tr.get_choices()["y1"]
        old_y2 = tr.get_choices()["y2"]
        old_target_density = genjax.normal.logpdf(
            old_y1, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y2, old_y1, 1.0)
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, _ = request.edit(tr, ())
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
            _ = genjax.normal(y1, 0.001) @ "y2"

        tr, _ = linked_normal.importance(C.kw(y2=3.0), ())
        request = Regenerate(S["y1"])

        # Run Metropolis-Hastings for 200 steps.
        for _ in range(500):
            new_tr, w, _, _ = request.edit(tr, ())
            check = jnp.log(genjax.uniform.sample(0.0, 1.0)) < w
            tr = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), new_tr, tr)

        assert tr.get_choices()["y1"] == pytest.approx(3.0, 8e-2)


class TestRejuvenate:
    def test_simple_normal_correctness(self):
        @genjax.gen
        def simple_normal():
            _ = genjax.normal(0.0, 1.0) @ "y1"

        tr = simple_normal.simulate(())
        old_v = tr.get_choices()["y1"]

        #####
        # Suppose the proposal is the prior, and its symmetric.
        #####

        request = DefRequest({
            "y1": Rejuvenate(
                genjax.normal,
                lambda chm: (0.0, 1.0),
            )
        })
        new_tr, w, _, _ = request.edit(tr, ())
        new_v = new_tr.get_choices()["y1"]
        assert old_v != new_v
        assert w == 0.0

    def test_linked_normal_rejuvenate_convergence(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 3.0) @ "y1"
            _ = genjax.normal(y1, 0.001) @ "y2"

        tr, _ = linked_normal.importance(C.kw(y2=3.0), ())

        request = DefRequest({
            "y1": Rejuvenate(
                genjax.normal,
                lambda chm: (chm.get_value(), 0.3),
            )
        })

        # Run Metropolis-Hastings for 100 steps.
        for _ in range(100):
            new_tr, w, _, _ = request.edit(tr, ())
            check = jnp.log(genjax.uniform.sample(0.0, 1.0)) < w
            tr = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), new_tr, tr)

        assert tr.get_choices()["y1"] == pytest.approx(3.0, 8e-2)


class TestHMC:
    def test_simple_normal_hmc(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 0.01) @ "y"
            return y

        tr, _ = model.importance(ChoiceMap.kw(y=3.0), ())
        request = HMC(Selection.at["x"], jnp.array(1e-2))
        editor = jax.jit(request.edit)

        # First, try moving x and test for correctness.
        old_x = tr.get_choices()["x"]
        old_y = tr.get_choices()["y"]
        old_target_density = genjax.normal.logpdf(
            old_x, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y, old_x, 0.01)
        new_tr, fwd_w, _, _ = editor(tr, ())
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
        for _ in range(100):
            new_tr, *_ = editor(new_tr, ())
        assert new_tr.get_choices()["x"] == pytest.approx(3.0, 1e-2)

    def test_simple_scan_hmc(self):
        @genjax.gen
        def kernel(z, scanned_in):
            z = genjax.normal(z, 1.0) @ "x"
            _ = genjax.normal(z, 0.001) @ "y"
            return z, None

        model = kernel.scan(n=10)
        vchm = ChoiceMap.empty().at["y"].set(3.0 * jnp.ones(10))
        tr, _ = model.importance(vchm, (0.0, None))
        request = HMC(Selection.at["x"], jnp.array(1e-3))
        editor = jax.jit(request.edit)
        new_tr = tr
        for _ in range(100):
            new_tr, *_ = editor(new_tr, Diff.no_change((0.0, None)))
        assert new_tr.get_choices()[:, "x"] == pytest.approx(3.0, 8e-2)

    @pytest.mark.skip(reason="needs more work")
    def test_hmm_hmc(self):
        @genjax.gen
        def simulate_motion_step(carry, scanned_in):
            (pos, pos_noise, obs_noise, dt) = carry
            new_latent_position = genjax.mv_normal_diag(pos, pos_noise) @ "pos"
            _ = genjax.mv_normal_diag(new_latent_position, obs_noise) @ "obs_pos"
            return (
                new_latent_position,
                pos_noise,
                obs_noise,
                dt,
            ), new_latent_position

        @genjax.gen
        def simple_hmm(position_noise, observation_noise, dt):
            initial_y_pos = genjax.normal(0.5, 0.01) @ "init_pos"
            initial_position = jnp.array([0.0, initial_y_pos])
            _ = (
                genjax.mv_normal_diag(initial_position, observation_noise)
                @ "init_obs_pos"
            )
            _, tracks = (
                simulate_motion_step.scan(n=10)(
                    (
                        initial_position,
                        position_noise,
                        observation_noise,
                        dt,
                    ),
                    None,
                )
                @ "tracks"
            )
            return jnp.vstack([initial_position, tracks])

        # Simulate ground truth from the model.
        ground_truth = simple_hmm.simulate(
            (jnp.array([1e-1, 1e-1]), jnp.array([1e-1, 1e-1]), 0.1),
        )

        # Create an initial importance sample.
        obs = ChoiceMap.empty()
        obs = obs.at["tracks", :, "obs_pos"].set(
            ground_truth.get_choices()["tracks", :, "obs_pos"]
        )
        obs = obs.at["init_obs_pos"].set(ground_truth.get_choices()["init_obs_pos"])
        init_tr, _ = simple_hmm.importance(
            obs,
            (jnp.array([1e-1, 1e-1]), jnp.array([1e-1, 1e-1]), 0.1),
        )

        def _rejuvenation(eps):
            def _inner(carry, _):
                (key, tr) = carry
                key, sub_key = jax.random.split(key)
                request = HMC(Selection.at["init_pos"], eps)
                new_tr, w, _, _ = request.edit(tr, Diff.no_change(tr.get_args()))
                check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
                tr = jtu.tree_map(
                    lambda v1, v2: jnp.where(check, v1, v2),
                    new_tr,
                    tr,
                )
                request = HMC(Selection.at["tracks", ..., "pos"], eps)
                key, sub_key = jax.random.split(key)
                new_tr, w, _, _ = request.edit(tr, Diff.no_change(tr.get_args()))
                key, sub_key = jax.random.split(key)
                check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
                tr = jtu.tree_map(
                    lambda v1, v2: jnp.where(check, v1, v2),
                    new_tr,
                    tr,
                )
                return (tr,), None

            return _inner

        def rejuvenation(length: int):
            def inner(key, tr, eps):
                (new_tr,), _ = jax.lax.scan(
                    _rejuvenation(eps),
                    (tr,),
                    length=length,
                )
                return new_tr

            return inner

        # Run MH with HMC.
        rejuvenator = jax.jit(rejuvenation(3000))
        new_tr = rejuvenator(init_tr, jnp.array(1e-4))
        assert init_tr.get_choices()["tracks", 0, "pos"] != pytest.approx(
            ground_truth.get_choices()["tracks", 0, "pos"], 1e-5
        )
        assert init_tr.get_choices()["tracks", -1, "pos"] != pytest.approx(
            ground_truth.get_choices()["tracks", -1, "pos"], 1e-5
        )
        assert new_tr.get_choices()["init_pos"] != pytest.approx(
            init_tr.get_choices()["init_pos"], 1e-5
        )
        assert new_tr.get_choices()["tracks", 0, "pos"] != pytest.approx(
            init_tr.get_choices()["tracks", 0, "pos"], 1e-5
        )
        assert new_tr.get_choices()["tracks", 0, "pos"] == pytest.approx(
            ground_truth.get_choices()["tracks", 0, "pos"], 5e-2
        )
        assert new_tr.get_choices()["tracks", -1, "pos"] == pytest.approx(
            ground_truth.get_choices()["tracks", -1, "pos"], 5e-2
        )

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

        tr, _ = model.importance(ChoiceMap.kw(y=3.0), ())
        request = DefRequest(
            {"x": SafeHMC(Selection.at["x"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        new_tr, w, *_ = editor(tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert w != 0.0

        # Compositional request _including_ HMC.
        request = DefRequest(
            {
                "x": SafeHMC(Selection.at["x"], jnp.array(1e-2)),
                "y": DefRequest({
                    "x": Regenerate(Selection.all()),
                    "y": Update(C.choice(3.0)),
                }),
            },
        )
        editor = jax.jit(request.edit)
        new_tr, w, *_ = editor(tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert new_tr.get_choices()["y", "x"] != tr.get_choices()["y", "x"]
        assert w != 0.0

        request = DefRequest(
            {"x": SafeHMC(Selection.at["y"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        with pytest.raises(Exception):
            new_tr, w, *_ = editor(tr, ())


class TestDiffCoercion:
    def test_diff_coercion(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            return y1 + y2

        tr = simple_normal.simulate(())

        # Test that DiffCoercion.edit is being
        # properly used compositionally.
        def assert_no_change(v):
            assert Diff.static_check_no_change(v)
            return v

        request = DefRequest({
            "y1": Regenerate(Selection.all()),
            "y2": DiffAnnotate(
                EmptyRequest(),
                argdiff_fn=assert_no_change,
            ),
        })

        with pytest.raises(Exception):
            request.edit(tr, ())

        # Test equivalent between requests which use
        # DiffCoercion in trivial ways.
        unwrapped_request = DefRequest({
            "y1": Regenerate(Selection.all()),
        })
        wrapped_request = DefRequest({
            "y1": Regenerate(Selection.all()).contramap(assert_no_change),
            "y2": EmptyRequest().map(assert_no_change),
        })
        _, w, _, _ = genjax.seed(unwrapped_request.edit)(jrand.key(1), tr, ())
        _, w_, _, _ = genjax.seed(wrapped_request.edit)(jrand.key(1), tr, ())
        assert w == w_

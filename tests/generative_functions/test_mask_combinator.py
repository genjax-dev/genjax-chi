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
import jax.tree_util as jtu
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax._src.generative_functions.combinators.vmap import VmapTrace


@genjax.mask
@genjax.gen
def model(x):
    z = genjax.normal(x, 1.0) @ "z"
    return z


class TestMaskCombinator:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_mask_simple_normal_true(self, key):
        tr = jax.jit(model.simulate)(key, (True, -4.0))
        assert tr.get_score() == tr.inner.get_score()
        assert tr.get_retval() == genjax.Mask(tr.inner.get_retval(), jnp.array(True))

        tr = jax.jit(model.simulate)(key, (False, -4.0))
        assert tr.get_score() == 0.0
        assert tr.get_retval() == genjax.Mask(tr.inner.get_retval(), jnp.array(False))

    def test_mask_simple_normal_false(self, key):
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        assert tr.get_score() == 0.0
        assert not tr.get_retval().flag

        score, retval = jax.jit(model.assess)(tr.get_choices(), tr.get_args())
        assert score == 0.0
        assert not retval.flag

        _, w = jax.jit(model.importance)(key, C["z"].set(-2.0), tr.get_args())
        assert w == 0.0

    def test_mask_update_weight_to_argdiffs_from_true(self, key):
        # pre-update, the mask is True
        tr = model.simulate(key, (True, 2.0))

        # mask check arg transition: True --> True
        argdiffs = (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1]
        assert w == 0.0
        # mask check arg transition: True --> False
        argdiffs = (Diff.unknown_change(False), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == -tr.get_score()

    def test_mask_update_weight_to_argdiffs_from_false(self, key):
        # pre-update mask arg is False
        tr = jax.jit(model.simulate)(key, (False, 2.0))

        # mask check arg transition: False --> True
        w = tr.update(
            key,
            C.n(),
            (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1])),
        )[1]
        assert w == tr.inner.update(key, C.n())[1] + tr.inner.get_score()
        assert w == tr.inner.update(key, C.n())[0].get_score()

        # mask check arg transition: False --> False
        w = tr.update(
            key,
            C.n(),
            (
                Diff.unknown_change(False),
                Diff.no_change(tr.get_args()[1]),
            ),
        )[1]
        assert w == 0.0
        assert w == tr.get_score()

    def test_mask_vmap(self, key):
        @genjax.gen
        def init():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        masks = jnp.array([True, False, True])

        @genjax.gen
        def model_2():
            vmask_init = init.mask().vmap(in_axes=(0))(masks) @ "init"
            return vmask_init

        tr = model_2.simulate(key, ())
        retval = tr.get_retval()
        retval_flag = retval.flag
        retval_val = retval.unmask()
        assert tr.get_score() == jnp.sum(
            retval_flag
            * jax.vmap(lambda v: genjax.normal.logpdf(v, 0.0, 1.0))(retval_val)
        )
        vmap_tr = tr.get_subtrace("init")
        assert isinstance(vmap_tr, VmapTrace)
        inner_scores = jax.vmap(lambda tr: tr.get_score())(vmap_tr.inner)
        # score should be sum of sub-scores masked True
        assert tr.get_score() == inner_scores[0] + inner_scores[2]

    def test_mask_update_weight_to_argdiffs_from_false_(self, key):
        # pre-update mask arg is False
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        # mask check arg transition: False --> True
        argdiffs = (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1] + tr.inner.get_score()
        assert w == tr.inner.update(key, C.n())[0].get_score()
        # mask check arg transition: False --> False
        argdiffs = (Diff.unknown_change(False), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == 0.0
        assert w == tr.get_score()

    def test_masked_iterate_final_update(self, key):
        masks = jnp.array([True, True])

        @genjax.gen
        def step(x):
            _ = (
                genjax.normal.mask().vmap(in_axes=(0, None, None))(masks, x, 1.0)
                @ "rats"
            )
            return x

        # Create some initial traces:
        key = jax.random.key(0)
        mask_steps = jnp.arange(10) < 5
        model = step.masked_iterate_final()
        init_particle = model.simulate(key, (0.0, mask_steps))

        assert jnp.array_equal(init_particle.get_retval(), jnp.array(0.0))

        step_particle, step_weight, _, _ = model.update(
            key, init_particle, C.n(), Diff.no_change((0.0, mask_steps))
        )
        assert jnp.array_equal(step_weight, jnp.array(0.0))
        assert jnp.array_equal(step_particle.get_retval(), jnp.array(0.0))

        # Testing inference working when we extend the model by unmasking a value.
        argdiffs_ = (Diff.no_change(0.0), Diff.unknown_change(jnp.arange(10) < 6))
        step_particle, step_weight, _, _ = model.update(
            key, init_particle, C.n(), argdiffs_
        )
        assert step_weight != jnp.array(0.0)
        assert step_particle.get_score() == step_weight + init_particle.get_score()

    def test_masked_iterate(self, key):
        masks = jnp.array([True, True])

        @genjax.gen
        def step(x):
            _ = (
                genjax.normal.mask().vmap(in_axes=(0, None, None))(masks, x, 1.0)
                @ "rats"
            )
            return x

        # Create some initial traces:
        key = jax.random.key(0)
        mask_steps = jnp.arange(10) < 5
        model = step.masked_iterate()
        init_particle = model.simulate(key, (0.0, mask_steps))
        assert jnp.array_equal(init_particle.get_retval(), jnp.zeros(11)), (
            "0.0 is threaded through 10 times in addition to the initial value"
        )

    def test_mask_scan_update_type_error(self, key):
        @genjax.gen
        def model_inside():
            masks = jnp.array([True, False, True])
            return genjax.normal(0.0, 1.0).mask().vmap()(masks) @ "init"

        outside_mask = jnp.array([True, False, True])

        @genjax.gen
        def model_outside():
            return genjax.normal(0.0, 1.0).mask().vmap()(outside_mask) @ "init"

        # These tests guard against regression to a strange case where it makes a difference whether
        # a constant `jnp.array` of flags is created inside or outside of a generative function.
        # When inside, the array is recast by JAX into a numpy array, since it appears in the
        # literal pool of a compiled function, but not when outside, where it escapes such
        # treatment.
        inside_tr = model_inside.simulate(key, ())
        outside_tr = model_outside.simulate(key, ())

        assert outside_tr.get_score() == inside_tr.get_score()
        assert jtu.tree_map(
            jnp.array_equal, inside_tr.get_retval(), outside_tr.get_retval()
        )
        assert jtu.tree_map(
            jnp.array_equal, inside_tr.get_choices(), outside_tr.get_choices()
        )

        retval = outside_tr.get_retval()
        retval_masks = retval.flag
        retval_value = retval.unmask()
        assert outside_tr.get_score() == jnp.sum(
            retval_masks
            * jax.vmap(lambda v: genjax.normal.logpdf(v, 0.0, 1.0))(retval_value)
        )

    def test_mask_fails_with_vector_mask(self, key):
        @genjax.gen
        def model():
            return genjax.normal(0.0, 1.0) @ "x"

        masks = jnp.array([True, True, False])

        def simulate_masked(key, masks):
            return model.mask().simulate(key, (masks,))

        with pytest.raises(TypeError):
            simulate_masked(key, masks)

        tr = model.mask().vmap().simulate(key, (masks,))

        # note that it's still possible to vmap.
        assert jnp.all(tr.get_retval().flag == masks)

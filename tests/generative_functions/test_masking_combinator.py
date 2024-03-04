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

import genjax
import jax
import jax.numpy as jnp


class TestMapCombinator:
    def test_mask_simple_normal_true(self):
        @genjax.masking_combinator
        @genjax.static_gen_fn
        def model(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(model.simulate)(key, (True, (-4.0,)))
        assert tr.get_score() == tr.inner.get_score()
        assert tr.get_retval() == genjax.Mask(jnp.array(True), tr.inner.get_retval())

        score, retval = jax.jit(model.assess)(tr.get_choices(), tr.get_args())
        assert score == tr.get_score()
        assert retval == tr.get_retval()

        _, w = jax.jit(model.importance)(
            key, genjax.Mask(True, genjax.choice_map(dict(z=-2.0))), tr.get_args()
        )
        assert (
            w
            == model.inner.importance(key, genjax.choice_map(dict(z=-2.0)), (-4.0,))[1]
        )

    def test_mask_simple_normal_false(self):
        @genjax.masking_combinator
        @genjax.static_gen_fn
        def model(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(model.simulate)(key, (False, (2.0,)))
        assert tr.get_score() == 0.0
        assert not tr.get_retval().flag

        score, retval = jax.jit(model.assess)(tr.get_choices(), tr.get_args())
        assert score == 0.0
        assert not retval.flag

        _, w = jax.jit(model.importance)(
            key, genjax.Mask(False, genjax.choice_map(dict(z=-2.0))), tr.get_args()
        )
        assert w == 0.0

    def test_mask_of_map(self):
        @genjax.map_combinator(in_axes=(0, (0,)))
        @genjax.masking_combinator
        @genjax.static_gen_fn
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        @genjax.map_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def unmasked_kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        mask1 = jnp.array([False, True, True, False, True])
        mask2 = jnp.array([True, True, True, False, False])

        tr = jax.jit(kernel.simulate)(key, (mask1, (jnp.arange(5.0),)))
        assert tr.get_score() == jnp.sum(tr.inner.inner.get_score()[mask1])
        assert (tr.get_retval().flag == mask1).all()

        score, retval = jax.jit(kernel.assess)(
            tr.get_choices(), (mask2, (jnp.arange(5.0),))
        )
        assert score == jnp.sum(tr.inner.inner.get_score()[mask2])
        assert (retval.flag == mask2).all()

        choice = genjax.vector_choice_map(
            genjax.Mask(
                mask2, genjax.choice_map(dict(z=jnp.array([0.5, 1.7, 0.0, 0.3, 3.0])))
            )
        )
        unmasked_choice = genjax.vector_choice_map(dict(z=jnp.array([0.5, 1.7, 0.0])))
        _, w = jax.jit(kernel.importance)(key, choice, (mask2, (jnp.arange(5.0),)))
        assert w == jnp.sum(
            unmasked_kernel.importance(key, unmasked_choice, (jnp.arange(3.0),))[1]
        )

    def test_masking_of_nonfinite_scores(self):
        @genjax.map_combinator(in_axes=(0, (None,)))
        @genjax.masking_combinator
        @genjax.static_gen_fn
        def f(scale):
            z = genjax.trace("z", genjax.normal)(0.0, scale)
            return z

        key = jax.random.PRNGKey(314159)
        mask = jnp.array([False, True, False, True, False])
        arg = jnp.array(1.0)

        choice = genjax.vector_choice_map(
            genjax.Mask(
                mask,
                genjax.choice_map(
                    dict(z=jnp.array([-jnp.inf, 1.7, jnp.inf, 0.3, jnp.nan]))
                ),
            )
        )

        score, retval = jax.jit(f.assess)(choice, (mask, (arg,)))
        assert jnp.isfinite(score)
        arg_grad = jax.grad(lambda a: f.assess(choice, (mask, (a,)))[0])(arg)
        # TODO: Fix gradient so that it is not NaN.
        # assert jnp.isfinite(arg_grad)
        del arg_grad

        tr, w = jax.jit(f.importance)(key, choice, (mask, (arg,)))
        assert (~jnp.isfinite(tr.inner.inner.get_score()[~mask])).all()
        assert jnp.isfinite(w)
        assert jnp.isfinite(tr.get_score())
        arg_grad = jax.grad(lambda a: f.importance(key, choice, (mask, (a,)))[1])(arg)
        # TODO: Fix gradient so that it is not NaN.
        # assert jnp.isfinite(arg_grad)
        del arg_grad

        # TODO(jburnim): Test MaskingTrace.project.

    def test_combinator(self):
        @genjax.masking_combinator
        @genjax.static_gen_fn
        def model():
            """Model docstring"""
            return genjax.normal(0.0, 1.0) @ "y"

        assert model.__doc__ == "Model docstring"

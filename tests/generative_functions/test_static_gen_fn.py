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

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMap, Diff, Pytree, Regenerate, StaticRequest, Update
from genjax import ChoiceMapBuilder as C
from genjax import Selection as S
from genjax._src.core.typing import Array
from genjax._src.generative_functions.static import MissingAddress
from genjax.generative_functions.static import AddressReuse
from genjax.typing import FloatArray

#############
# Datatypes #
#############

##################################
# Generative function interfaces #
##################################


class TestStaticGenFnMetadata:
    def test_docstring_transfer(self):
        def original_function(x: float, y: float) -> float:
            """
            This is a test function that adds two numbers.

            Args:
                x (float): The first number
                y (float): The second number

            Returns:
                float: The sum of x and y
            """
            return x + y

        wrapped_function = genjax.gen(original_function)

        assert wrapped_function.__doc__ == original_function.__doc__
        assert wrapped_function.__name__ == original_function.__name__
        assert wrapped_function.__module__ == original_function.__module__
        assert wrapped_function.__qualname__ == original_function.__qualname__
        assert getattr(wrapped_function, "__wrapped__") == original_function

    def test_docstring_transfer_with_annotations(self):
        @genjax.gen
        def annotated_function(x: float, y: float) -> float:
            """
            This is an annotated test function that multiplies two numbers.

            Args:
                x (float): The first number
                y (float): The second number

            Returns:
                float: The product of x and y
            """
            return x * y

        assert annotated_function.__doc__ is not None
        assert "This is an annotated test function" in annotated_function.__doc__
        assert annotated_function.__annotations__ == {
            "x": float,
            "y": float,
            "return": float,
        }


class TestMisc:
    def test_switch_chm_and_static(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(0.0, 1.0) @ "y"
            return x, y

        switch_chm = ChoiceMap.switch(jnp.int_(1), [C["x"].set(2.3), C["x"].set(3.4)])
        key = jax.random.key(0)

        switch_and_y = switch_chm.merge(C["y"].set(4.5))

        tr, _ = model.importance(key, switch_and_y, ())

        assert tr.get_retval() == (3.4, 4.5)

    def test_assess_vmap_masked(self):
        """
        Test case provided by George Matheos in GEN-903.
        """
        gf = genjax.flip.vmap(in_axes=(0,))

        @jax.jit
        def get_choicemap(idx):
            return genjax.ChoiceMap.switch(
                idx=idx,
                chms=[
                    C.set(jnp.array([0, 0, 1], dtype=bool)),
                    C.set(jnp.array([1, 1, 1], dtype=bool)),
                ],
            )

        chm = get_choicemap(1)

        flipprobs = jnp.array([0.2, 0.4, 0.6])
        tr, w = gf.importance(jax.random.key(0), chm, (flipprobs,))
        # ^ This line runs.
        # However, when I try to call assess in the exact
        # same configuration, I get an error.
        score, r = gf.assess(chm, (flipprobs,))
        assert jnp.array_equal(tr.get_retval(), r)
        assert tr.get_score() == score
        assert score == w, "no weight change w/ same chm"

    def test_static_retval(self):
        """
        Test that it's possible return a literal from a generative function. and successfully call update on such a function.
        """

        @genjax.gen
        def f():
            return 1

        k = jax.random.key(0)
        tr = f.simulate(k, ())
        tr.update(k, C.n(), ())
        assert tr.get_retval() == 1

    def test_get_zero_trace(self):
        @genjax.gen
        def model(x):
            y = genjax.normal(x, 1.0) @ "y"
            z = genjax.bernoulli(probs=0.7) @ "z"
            return y + z

        zero_trace = model.get_zero_trace(0.0)

        assert isinstance(zero_trace, genjax.Trace)
        assert zero_trace.get_args() == (0.0,)
        assert zero_trace.get_retval() == 0.0
        assert zero_trace.get_score() == 0.0

        zero_choices = zero_trace.get_choices()
        assert "y" in zero_choices
        assert "z" in zero_choices
        assert zero_choices["y"] == 0.0
        assert zero_choices["z"] == 0.0

    def test_get_zero_trace_with_nested_structure(self):
        @genjax.gen
        def nested_model():
            @genjax.gen
            def inner_model():
                return genjax.normal(0.0, 1.0) @ "inner"

            outer = genjax.normal(0.0, 1.0) @ "outer"
            inner_result = inner_model() @ "nested"
            return outer + inner_result

        zero_trace = nested_model.get_zero_trace()

        assert isinstance(zero_trace, genjax.Trace)
        assert zero_trace.get_args() == ()
        assert zero_trace.get_retval() == 0.0
        assert zero_trace.get_score() == 0.0

        zero_choices = zero_trace.get_choices()
        assert zero_choices["outer"] == 0.0
        assert zero_choices["nested", "inner"] == 0.0


class TestStaticGenFnSimulate:
    def test_simulate_with_no_choices(self):
        @genjax.gen
        def empty(x):
            return jnp.square(x - 3.0)

        key = jax.random.key(314159)
        fn = jax.jit(empty.simulate)
        key, sub_key = jax.random.split(key)
        tr = fn(sub_key, (jnp.ones(4),))
        assert tr.get_score() == 0.0

    def test_simple_normal_simulate(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        fn = jax.jit(simple_normal.simulate)
        key, sub_key = jax.random.split(key)
        tr = fn(sub_key, ())
        choice = tr.get_choices()
        (_, score1) = genjax.normal.importance(key, choice.get_submap("y1"), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, choice.get_submap("y2"), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_multiple_returns(self):
        @genjax.gen
        def simple_normal_multiple_returns():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr.get_choices()["y1"]
        y2_ = tr.get_choices()["y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (score1, _) = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        (score2, _) = genjax.normal.assess(C.v(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_hierarchical_simple_normal_multiple_returns(self):
        @genjax.gen
        def _submodel():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        @genjax.gen
        def hierarchical_simple_normal_multiple_returns():
            y1, y2 = _submodel() @ "y1"
            return y1, y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(hierarchical_simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr.get_choices()["y1", "y1"]
        y2_ = tr.get_choices()["y1", "y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (score1, _) = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        (score2, _) = genjax.normal.assess(C.v(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)


class TestStaticGenFnAssess:
    def test_assess_with_no_choices(self):
        @genjax.gen
        def empty(x):
            return jnp.square(x - 3.0)

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(empty.simulate)(sub_key, (jnp.ones(4),))
        jitted = jax.jit(empty.assess)
        chm = tr.get_choices()
        (score, _retval) = jitted(chm, (jnp.ones(4),))
        assert score == tr.get_score()

    def test_simple_normal_assess(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_normal.assess)
        choice = tr.get_choices()
        (score, _retval) = jitted(choice, ())
        assert score == tr.get_score()

    def test_assess_missing_address(self):
        @genjax.gen
        def model():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        with pytest.raises(MissingAddress) as exc:
            _ = model.assess(C["y1"].set(1.0), ())
        assert exc.value.args == ("y2",)

        with pytest.raises(MissingAddress) as exc:
            _ = model.assess(C["y2"].set(1.0), ())
        assert exc.value.args == ("y1",)

        score_retval = model.assess(C["y1"].set(1.0).at["y2"].set(-1.0), ())
        assert score_retval == (-2.837877, 0.0)


@Pytree.dataclass
class CustomTree(genjax.Pytree):
    x: Any
    y: Any


@genjax.gen
def simple_normal(custom_tree):
    y1 = genjax.normal(custom_tree.x, 1.0) @ "y1"
    y2 = genjax.normal(custom_tree.y, 1.0) @ "y2"
    return CustomTree(y1, y2)


@Pytree.dataclass
class _CustomNormal(genjax.Distribution[Array]):
    def estimate_logpdf(self, key, v, *args):
        v, custom_tree = args
        w, _ = genjax.normal.assess(v, (custom_tree.x, custom_tree.y))
        return w

    def random_weighted(self, key, *args):
        (custom_tree,) = args
        return genjax.normal.random_weighted(key, custom_tree.x, custom_tree.y)


CustomNormal = _CustomNormal()


@genjax.gen
def custom_normal(custom_tree):
    y = CustomNormal(custom_tree) @ "y"
    return CustomTree(y, y)


class TestStaticGenFnCustomPytree:
    def test_simple_normal_simulate(self):
        key = jax.random.key(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(simple_normal.simulate)
        tr = fn(key, (init_tree,))
        choice = tr.get_choices()
        (_, score1) = genjax.normal.importance(
            key, choice.get_submap("y1"), (init_tree.x, 1.0)
        )
        (_, score2) = genjax.normal.importance(
            key, choice.get_submap("y2"), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_custom_normal_simulate(self):
        key = jax.random.key(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(custom_normal.simulate)
        tr = fn(key, (init_tree,))
        choice = tr.get_choices()
        (_, score) = genjax.normal.importance(
            key, choice.get_submap("y"), (init_tree.x, init_tree.y)
        )
        test_score = score
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_importance(self):
        key = jax.random.key(314159)
        init_tree = CustomTree(3.0, 5.0)
        choice = C["y1"].set(5.0)
        fn = jax.jit(simple_normal.importance)
        (tr, w) = fn(key, choice, (init_tree,))
        choice = tr.get_choices()
        (_, score1) = genjax.normal.importance(
            key, choice.get_submap("y1"), (init_tree.x, 1.0)
        )
        (_, score2) = genjax.normal.importance(
            key, choice.get_submap("y2"), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
        assert w == pytest.approx(score1, 0.01)


class TestStaticGenFnGradients:
    def test_simple_normal_assess(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        tr = jax.jit(simple_normal.simulate)(key, ())
        jitted = jax.jit(simple_normal.assess)
        choice = tr.get_choices()
        (score, _) = jitted(choice, ())
        assert score == tr.get_score()


class TestStaticGenFnImportance:
    def test_importance_simple_normal(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        fn = simple_normal.importance
        choice = C["y1"].set(0.5).at["y2"].set(0.5)
        key, sub_key = jax.random.split(key)
        (out, _) = fn(sub_key, choice, ())
        (_, score_1) = genjax.normal.importance(
            key, choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score_2) = genjax.normal.importance(
            key, choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert choice["y1"] == out.get_choices()["y1"]
        assert choice["y2"] == out.get_choices()["y2"]
        assert out.get_score() == pytest.approx(test_score, 0.01)

    def test_importance_weight_correctness(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        # Full constraints.
        key = jax.random.key(314159)
        choice = C["y1"].set(0.5).at["y2"].set(0.5)
        (tr, w) = simple_normal.importance(key, choice, ())
        y1 = tr.get_choices()["y1"]
        y2 = tr.get_choices()["y2"]
        assert y1 == 0.5
        assert y2 == 0.5
        (_, score_1) = genjax.normal.importance(
            key, choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score_2) = genjax.normal.importance(
            key, choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(test_score, 0.0001)

        # Partial constraints.
        choice = C["y2"].set(0.5)
        (tr, w) = simple_normal.importance(key, choice, ())
        tr_chm = tr.get_choices()
        y1 = tr_chm.get_submap("y1")
        y2 = tr_chm.get_submap("y2")
        assert tr_chm["y2"] == 0.5
        score_1, _ = genjax.normal.assess(y1, (0.0, 1.0))
        score_2, _ = genjax.normal.assess(y2, (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(score_2, 0.0001)

        # No constraints.
        choice = C.n()
        (tr, w) = simple_normal.importance(key, choice, ())
        tr_chm = tr.get_choices()
        y1 = tr_chm.get_submap("y1")
        y2 = tr_chm.get_submap("y2")
        score_1, _ = genjax.normal.assess(y1, (0.0, 1.0))
        score_2, _ = genjax.normal.assess(y2, (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == 0.0


####################################################
#          Remember: the update weight math        #
#                                                  #
#   log p(r′,t′;x′) + log q(r;x,t) - log p(r,t;x)  #
#       - log q(r′;x′,t′) - q(t′;x′,t+u)           #
#                                                  #
####################################################


class TestStaticGenFnUpdate:
    def test_simple_normal_update(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_normal.update)

        new = C["y1"].set(2.0)
        original_choice = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(
            sub_key,
            tr,
            new,
            (),
        )

        updated_choice = updated.get_choices()
        y1 = updated_choice["y1"]
        _y2 = updated_choice["y2"]
        (_, score1) = genjax.normal.importance(
            key, updated_choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score2) = genjax.normal.importance(
            key, updated_choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert original_choice["y1",] == discard["y1",]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = C["y1"].set(2.0).at["y2"].set(3.0)
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())
        updated_choice = updated.get_choices()
        y1 = updated_choice.get_submap("y1")
        y2 = updated_choice.get_submap("y2")
        (_, score1) = genjax.normal.importance(key, y1, (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, y2, (0.0, 1.0))
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_linked_normal_update(self):
        @genjax.gen
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_linked_normal.update)

        new = C["y1"].set(2.0)
        original_choice = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())

        updated_choice = updated.get_choices()
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2"]
        y3 = updated_choice.get_submap("y3")
        score1, _ = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        score2, _ = genjax.normal.assess(C.v(y2), (y1, 1.0))
        score3, _ = genjax.normal.assess(y3, (y1 + y2, 1.0))
        test_score = score1 + score2 + score3
        assert original_choice["y1"] == discard["y1"]
        assert updated.get_score() == pytest.approx(original_score + w, 0.01)
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_hierarchical_normal(self):
        @genjax.gen
        def _inner(x):
            y1 = genjax.normal(x, 1.0) @ "y1"
            return y1

        @genjax.gen
        def simple_hierarchical_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = _inner(y1) @ "y2"
            y3 = _inner(y1 + y2) @ "y3"
            return y1 + y2 + y3

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_hierarchical_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_hierarchical_normal.update)

        new = C["y1"].set(2.0)
        original_choice = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)

        (updated, w, _, discard) = jitted(sub_key, tr, new, ())

        updated_choice = updated.get_choices()
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2", "y1"]
        y3 = updated_choice["y3", "y1"]
        assert y1 == new["y1"]
        assert y2 == original_choice["y2", "y1"]
        assert y3 == original_choice["y3", "y1"]
        score1, _ = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        score2, _ = genjax.normal.assess(C.v(y2), (y1, 1.0))
        score3, _ = genjax.normal.assess(C.v(y3), (y1 + y2, 1.0))
        test_score = score1 + score2 + score3
        assert original_choice["y1"] == discard["y1"]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def update_weight_correctness_general_assertions(self, simple_linked_normal):
        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_linked_normal.update)

        old_y1 = tr.get_choices()["y1"]
        old_y2 = tr.get_choices()["y2"]
        old_y3 = tr.get_choices()["y3"]
        new_y1 = 2.0
        new = C["y1"].set(new_y1)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(sub_key, tr, new, ())
        (_, w_edit, _, _) = tr.edit(sub_key, Update(new))
        assert w_edit == w

        # TestStaticGenFn weight correctness.
        updated_sample = updated.get_choices()
        assert updated_sample["y1"] == new_y1

        δ_y3 = (
            genjax.normal.assess(C.v(old_y3), (new_y1 + old_y2, 1.0))[0]
            - genjax.normal.assess(C.v(old_y3), (old_y1 + old_y2, 1.0))[0]
        )
        δ_y2 = (
            genjax.normal.assess(C.v(old_y2), (new_y1, 1.0))[0]
            - genjax.normal.assess(C.v(old_y2), (old_y1, 1.0))[0]
        )
        δ_y1 = (
            genjax.normal.assess(C.v(new_y1), (0.0, 1.0))[0]
            - genjax.normal.assess(C.v(old_y1), (0.0, 1.0))[0]
        )
        assert w == pytest.approx((δ_y3 + δ_y2 + δ_y1), 0.0001)

        # TestStaticGenFn composition of update calls.
        new_y3 = 2.0
        new = C["y3"].set(new_y3)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(sub_key, updated, new, ())
        assert updated.get_choices()["y3"] == 2.0
        correct_w = (
            genjax.normal.assess(C.v(new_y3), (new_y1 + old_y2, 1.0))[0]
            - genjax.normal.assess(C.v(old_y3), (new_y1 + old_y2, 1.0))[0]
        )
        assert w == pytest.approx(correct_w, 0.0001)

    def test_update_weight_correctness(self):
        @genjax.gen
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        # easy case
        self.update_weight_correctness_general_assertions(simple_linked_normal)

        @genjax.gen
        def curried_linked_normal(v1, v2, v3):
            y1 = genjax.normal(0.0, v1) @ "y1"
            y2 = genjax.normal(y1, v2) @ "y2"
            y3 = genjax.normal(y1 + y2, v3) @ "y3"
            return y1 + y2 + y3

        # curry
        self.update_weight_correctness_general_assertions(
            curried_linked_normal.partial_apply(1.0, 1.0, 1.0)
        )

        # double-curry
        self.update_weight_correctness_general_assertions(
            curried_linked_normal.partial_apply(1.0).partial_apply(1.0, 1.0)
        )

        @Pytree.dataclass
        class Model(Pytree):
            v1: Array
            v2: Array

            @genjax.gen
            def run(self, v3):
                y1 = genjax.normal(0.0, self.v1) @ "y1"
                y2 = genjax.normal(y1, self.v2) @ "y2"
                y3 = genjax.normal(y1 + y2, v3) @ "y3"
                return y1 + y2 + y3

        # model method
        m = Model(jnp.asarray(1.0), jnp.asarray(1.0))
        self.update_weight_correctness_general_assertions(m.run.partial_apply(1.0))

        @genjax.gen
        def m_linked(m: Model, v2, v3):
            y1 = genjax.normal(0.0, m.v1) @ "y1"
            y2 = genjax.normal(y1, v2) @ "y2"
            y3 = genjax.normal(y1 + y2, v3) @ "y3"
            return y1 + y2 + y3

        self.update_weight_correctness_general_assertions(
            m_linked.partial_apply(m).partial_apply(1.0, 1.0)
        )

        @genjax.gen
        def m_created_internally(scale: Array):
            m_internal = Model(scale, scale)
            return m_internal.run.inline(scale)

        self.update_weight_correctness_general_assertions(
            m_created_internally.partial_apply(jnp.asarray(1.0))
        )

    def test_update_pytree_argument(self):
        @Pytree.dataclass
        class SomePytree(genjax.Pytree):
            x: float | FloatArray
            y: float | FloatArray

        @genjax.gen
        def simple_linked_normal_with_tree_argument(tree):
            y1 = genjax.normal(tree.x, tree.y) @ "y1"
            return y1

        key = jax.random.key(314159)
        init_tree = SomePytree(0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal_with_tree_argument.simulate)(
            sub_key, (init_tree,)
        )
        jitted = jax.jit(simple_linked_normal_with_tree_argument.update)
        new_y1 = 2.0
        constraints = C["y1"].set(new_y1)
        key, sub_key = jax.random.split(key)
        (updated, _w, _, _) = jitted(
            sub_key,
            tr,
            constraints,
            (Diff.no_change(init_tree),),
        )
        assert updated.get_choices()["y1"] == new_y1
        new_tree = SomePytree(1.0, 2.0)
        key, sub_key = jax.random.split(key)
        (updated, _w, _, _) = jitted(
            sub_key,
            tr,
            constraints,
            (Diff.unknown_change(new_tree),),
        )
        assert updated.get_choices()["y1"] == new_y1


#####################
# Language features #
#####################


class TestStaticGenFnStaticAddressChecks:
    def test_simple_normal_addr_dup(self):
        @genjax.gen
        def simple_normal_addr_dup():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y1"
            return y1 + y2

        key = jax.random.key(314159)
        with pytest.raises(AddressReuse) as exc_info:
            _ = simple_normal_addr_dup.simulate(key, ())
        assert exc_info.value.args[0] == "y1"

    def test_simple_normal_addr_tracer(self):
        @genjax.gen
        def simple_normal_addr_tracer():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ y1
            return y1 + y2

        key = jax.random.key(314159)
        with pytest.raises(TypeError) as _:
            _ = simple_normal_addr_tracer.simulate(key, ())


class TestStaticGenFnForwardRef:
    def test_forward_ref(self):
        def make_gen_fn():
            @genjax.gen
            def proposal(x):
                x = outlier(x) @ "x"
                return x

            @genjax.gen
            def outlier(prob):
                is_outlier = genjax.bernoulli(probs=prob) @ "is_outlier"
                return is_outlier

            return proposal

        key = jax.random.key(314159)
        proposal = make_gen_fn()
        tr = proposal.simulate(key, (0.3,))

        assert tr.get_score() == genjax.bernoulli.logpdf(tr.get_retval(), probs=0.3)


class TestGenFnClosure:
    def test_gen_fn_closure(self):
        @genjax.gen
        def model():
            return genjax.normal(1.0, 0.001) @ "x"

        gfc = model()
        tr = gfc.simulate(jax.random.key(0), ())
        assert tr.get_score() == genjax.normal.logpdf(tr.get_retval(), 1.0, 0.001)
        # This failed in GEN-420
        tr_u, w = gfc.importance(jax.random.key(1), C.kw(x=1.1), ())
        assert tr_u.get_score() == genjax.normal.logpdf(tr_u.get_retval(), 1.0, 0.001)
        assert w == tr_u.get_score()

    def test_gen_fn_closure_with_kwargs(self):
        @genjax.gen
        def model(x, y, z: FloatArray | None = None):
            if z is None:
                raise ValueError("z must be provided")

            _sampled = genjax.normal(x + y, z) @ "sampled"
            return z

        key = jax.random.key(0)

        # show that we error with z missing:
        with pytest.raises(ValueError, match="z must be provided"):
            model(1.0, 2.0)(key)

        gfc = model(1.0, 2.0, z=3.0)

        # the keyword args are passed through:
        assert gfc(key) == 3.0

        # we can override z:
        assert gfc(key, z=10.0) == 10.0

        # handle_kwargs doesn't affect function calls:
        assert gfc.handle_kwargs()(key, z=5.0) == gfc(key, z=5.0)

        # Test simulate with kwargs
        arg_tuple = (1.0, 2.0, 3.0)
        assert (
            gfc.simulate(key, ()).get_choices()
            == model.simulate(key, arg_tuple).get_choices()
        )

        # Test assess with kwargs
        chm = C.kw(sampled=3.5)
        assert gfc.assess(chm, ()) == model.assess(chm, arg_tuple)

        # Test generate with kwargs
        constraint = C.kw(sampled=3.0)
        assert (
            gfc.importance(key, constraint, ())[1]
            == model.generate(key, constraint, arg_tuple)[1]
        )

        # Test __abstract_call__ with kwargs
        assert gfc.__abstract_call__() == model.__abstract_call__(*arg_tuple)

        # handle_kwargs is idempotent on a gfc
        assert gfc.handle_kwargs() == gfc.handle_kwargs().handle_kwargs()


class TestStaticEditRequest:
    def test_static_edit_request_composition(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(0)
        tr = simple_normal.simulate(key, ())
        request = StaticRequest({
            "y1": Regenerate(S.all()),
            "y2": Update(C.v(3.0)),
        })
        key, sub_key = jax.random.split(key)
        new_tr, w, _, bwd_request = request.edit(key, tr, ())
        assert new_tr.get_choices()["y2"] == 3.0
        assert w != 0.0
        old_tr, w_, _, _ = bwd_request.edit(sub_key, new_tr, ())
        assert old_tr.get_choices()["y2"] == tr.get_choices()["y2"]
        assert w_ != 0.0
        assert w + w_ == 0.0

    def test_static_edit_request_with_tuple_addr(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ ("y1", "y3")
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.key(0)
        tr = simple_normal.simulate(key, ())
        request = StaticRequest({
            ("y1", "y3"): Regenerate(S.all()),
            "y2": Update(C.v(3.0)),
        })
        key, sub_key = jax.random.split(key)
        new_tr, w, _, bwd_request = request.edit(key, tr, ())
        assert new_tr.get_choices()["y2"] == 3.0
        assert w != 0.0
        old_tr, w_, _, _ = bwd_request.edit(sub_key, new_tr, ())
        assert old_tr.get_choices()["y2"] == tr.get_choices()["y2"]
        assert w_ != 0.0
        assert w + w_ == 0.0

    def test_static_edit_request_hierarchical(self):
        @genjax.gen
        def submodel():
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y2

        @genjax.gen
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ ("y1", "y3")
            y2 = submodel() @ "y2"
            return y1 + y2

        key = jax.random.key(0)
        tr = simple_normal.simulate(key, ())
        request = StaticRequest({
            ("y1", "y3"): Regenerate(S.all()),
            "y2": StaticRequest({"y2": Update(C.v(3.0))}),
        })
        key, sub_key = jax.random.split(key)
        new_tr, w, _, bwd_request = request.edit(key, tr, ())
        assert new_tr.get_choices()["y2", "y2"] == 3.0
        assert w != 0.0
        old_tr, w_, _, _ = bwd_request.edit(sub_key, new_tr, ())
        assert old_tr.get_choices()["y2", "y2"] == tr.get_choices()["y2", "y2"]
        assert w_ != 0.0
        assert w + w_ == 0.0


class TestHandleKwargs:
    def test_handle_kwargs(self):
        @genjax.gen
        def model(x, y, z: FloatArray | None = None):
            if z is None:
                raise ValueError("z must be provided")

            _sampled = genjax.normal(x + y, z) @ "sampled"
            return z

        # handle_kwargs produces a new model capable of handling keyword args.
        kwm = model.handle_kwargs()

        key = jax.random.key(0)
        kwm_tr = kwm.simulate(key, ((1.0,), {"y": 2.0, "z": 3.0}))
        model_tr = model.simulate(key, (1.0, 2.0, 3.0))

        assert kwm_tr.get_choices() == model_tr.get_choices()
        assert kwm_tr.get_score() == model_tr.get_score()
        assert kwm_tr.get_retval() == model_tr.get_retval()

        assert kwm_tr.get_args() == ((1.0,), {"y": 2.0, "z": 3.0})
        assert model_tr.get_args() == (1.0, 2.0, 3.0)


class TestStaticGenFnInline:
    def test_inline_simulate(self):
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

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        choices = tr.get_choices()
        assert "y1" in choices
        assert "y2" in choices
        tr = jax.jit(higher_higher_model.simulate)(key, ())
        choices = tr.get_choices()
        assert "y1" in choices
        assert "y2" in choices

    def test_inline_importance(self):
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

        key = jax.random.key(314159)
        choice = C["y1"].set(3.0)
        key, sub_key = jax.random.split(key)
        (tr, w) = jax.jit(higher_model.importance)(sub_key, choice, ())
        choices = tr.get_choices()
        assert w == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
        (tr, w) = jax.jit(higher_higher_model.importance)(key, choice, ())
        choices = tr.get_choices()
        assert w == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]

    def test_inline_update(self):
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

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        choice = C["y1"].set(3.0)
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        old_value = tr.get_choices().get_submap("y1")
        key, sub_key = jax.random.split(key)
        (tr, w, _rd, _) = jax.jit(higher_model.update)(sub_key, tr, choice, ())
        choices = tr.get_choices()
        assert (
            w
            == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
            - genjax.normal.assess(old_value, (0.0, 1.0))[0]
        )
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_higher_model.simulate)(sub_key, ())
        old_value = tr.get_choices().get_submap("y1")
        (tr, w, _rd, _) = jax.jit(higher_higher_model.update)(key, tr, choice, ())
        choices = tr.get_choices()
        assert w == pytest.approx(
            genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
            - genjax.normal.assess(old_value, (0.0, 1.0))[0],
            0.0001,
        )

    def test_inline_assess(self):
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

        _key = jax.random.key(314159)
        choice = C["y1"].set(3.0).at["y2"].set(3.0)
        (score, _ret) = jax.jit(higher_model.assess)(choice, ())
        assert (
            score
            == genjax.normal.assess(choice.get_submap("y1"), (0.0, 1.0))[0]
            + genjax.normal.assess(choice.get_submap("y2"), (0.0, 1.0))[0]
        )
        (score, _ret) = jax.jit(higher_higher_model.assess)(choice, ())
        assert (
            score
            == genjax.normal.assess(choice.get_submap("y1"), (0.0, 1.0))[0]
            + genjax.normal.assess(choice.get_submap("y2"), (0.0, 1.0))[0]
        )

    def test_gen_method(self):
        @Pytree.dataclass
        class Model(Pytree):
            foo: Array
            bar: Array

            @genjax.gen
            def run(self, x):
                y = genjax.normal(self.foo, self.bar) @ "y"
                z = genjax.normal(x, 1.0) @ "z"
                return y + z

        key = jax.random.key(0)
        # outside(1.0)(key)

        m = Model(jnp.asarray(4.0), jnp.asarray(6.0))
        tr = m.run.simulate(key, (1.0,))
        chm = tr.get_choices()

        assert tr.get_args() == (1.0,), (
            "The curried `self` arg is not present in get_args()"
        )

        assert tr.gen_fn.partial_args[0] == m, (
            "`self` is retrievable using `partial_args"
        )

        assert "y" in chm
        assert "z" in chm
        assert "q" not in chm

    def test_partial_apply(self):
        @genjax.gen
        def model(x, y, z):
            return genjax.normal(x, y + z) @ "x"

        double_curry = model.partial_apply(1.0).partial_apply(1.0)
        key = jax.random.key(0)

        tr = double_curry.simulate(key, (2.0,))
        assert tr.get_args() == (2.0,), (
            "both curried args are not present alongside the final arg"
        )

        assert tr.gen_fn.partial_args == (
            1.0,
            1.0,
        ), "They are present as `partial_args`"

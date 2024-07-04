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

import genjax
import jax
import jax.numpy as jnp
import pytest
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import SelectionBuilder as S
from genjax import UpdateProblemBuilder as U


@genjax.iterate(n=10)
@genjax.gen
def scanner(x):
    z = genjax.normal(x, 1.0) @ "z"
    return z


class TestIterateSimpleNormal:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_iterate_simple_normal(self, key):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key, sub_key = jax.random.split(key)
        tr = jax.jit(scanner.simulate)(sub_key, (0.01,))
        scan_score = tr.get_score()
        sel = S[..., "z"]
        assert tr.project(key, sel) == scan_score

    def test_iterate_simple_normal_importance(self):
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            assert tr.get_sample()[i, "z"].unmask() == 0.5
            value = tr.get_sample()[i, "z"].unmask()
            prev = tr.get_sample()[i - 1, "z"].unmask()
            assert w == genjax.normal.assess(C.v(value), (prev, 1.0))[0]

    def test_iterate_simple_normal_update(self):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, _w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            new_tr, _w, _rd, _bwd_problem = jax.jit(scanner.update)(
                sub_key,
                tr,
                U.g(
                    Diff.no_change((0.01,)),
                    C[i, "z"].set(1.0),
                ),
            )
            assert new_tr.get_sample()[i, "z"].unmask() == 1.0


@genjax.gen
def inc(prev):
    return prev + 1


@genjax.gen
def inc_tupled(arg):
    """Takes a pair, returns a pair."""
    prev, offset = arg
    return (prev + offset, offset)


class TestIterate:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_inc(self, key):
        """Baseline test that `inc` works!"""
        result = inc.simulate(key, (0,)).get_retval()
        assert result == 1

    def test_iterate(self, key):
        """
        `iterate` returns a generative function that applies the original
        function `n` times and returns an array of each result (not including
        the initial value).
        """
        result = inc.iterate(n=4).simulate(key, (0,)).get_retval()
        assert jnp.array_equal(result, jnp.array([1, 2, 3, 4]))

    def test_iterate_final(self, key):
        """
        `iterate_final` returns a generative function that applies the original
        function `n` times and returns the final result.
        """

        result = inc.iterate_final(n=10).simulate(key, (0,)).get_retval()
        assert jnp.array_equal(result, 10)

    def test_inc_tupled(self, key):
        """Baseline test demonstrating `inc_tupled`."""
        result = inc_tupled.simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(result, (2, 2))

    def test_iterate_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation.
        """
        result = inc_tupled.iterate(n=4).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(
            result, (jnp.array([2, 4, 6, 8]), jnp.array([2, 2, 2, 2]))
        )

    def test_iterate_final_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation. Same idea as above, but with
        `iterate_final`.
        """
        result = inc_tupled.iterate_final(n=10).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(result, (20, 2))


@genjax.gen
def add(carry, x):
    return carry + x


@genjax.gen
def add_tupled(acc, x):
    """accumulator state is a pair."""
    carry, offset = acc
    return (carry + x + offset, offset)


class TestScanFoldMethods:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_add(self, key):
        """Baseline test that `add` works!"""
        result = add.simulate(key, (0, 2)).get_retval()
        assert result == 2

    def test_scan(self, key):
        """
        `scan` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns an array of each intermediate accumulator value seen (not including the initial value).
        """
        result = add.accumulate().simulate(key, (0, jnp.ones(4))).get_retval()
        assert jnp.array_equal(result, jnp.array([1, 2, 3, 4]))

    def test_fold(self, key):
        """
        `fold` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns the final `accumulator` produces by folding in each element of `jnp.array(v)`.
        """

        result = add.reduce().simulate(key, (0, jnp.ones(10))).get_retval()
        assert jnp.array_equal(result, 10)

    def test_add_tupled(self, key):
        """Baseline test demonstrating `add_tupled`."""
        result = add_tupled.simulate(key, ((0, 2), 10)).get_retval()
        assert jnp.array_equal(result, (12, 2))

    def test_scan_tupled(self, key):
        """
        `scan` on function with tupled carry state works correctly.
        """
        result = (
            add_tupled.accumulate().simulate(key, ((0, 2), jnp.ones(4))).get_retval()
        )
        assert jnp.array_equal(
            result, (jnp.array([3, 6, 9, 12]), jnp.array([2, 2, 2, 2]))
        )

    def test_fold_tupled(self, key):
        """
        `fold` on function with tupled carry state works correctly.
        """
        result = add_tupled.reduce().simulate(key, ((0, 2), jnp.ones(10))).get_retval()
        assert jnp.array_equal(result, (30, 2))

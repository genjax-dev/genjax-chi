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

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff, IndexRequest, Regenerate, StaticRequest
from genjax import Selection as S
from genjax._src.core.typing import ArrayLike
from genjax.typing import FloatArray


@genjax.iterate(n=10)
@genjax.gen
def scanner(x):
    z = genjax.normal(x, 1.0) @ "z"
    return z


class TestIterateSimpleNormal:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_iterate_simple_normal(self, key):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key, sub_key = jax.random.split(key)
        tr = jax.jit(scanner.simulate)(sub_key, (0.01,))
        scan_score = tr.get_score()
        sel = genjax.Selection.all()
        assert tr.project(key, sel) == scan_score

    def test_iterate_simple_normal_importance(self, key):
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            value = tr.get_sample()[i, "z"]
            assert value == 0.5
            prev = tr.get_sample()[i - 1, "z"]
            assert w == genjax.normal.assess(C.v(value), (prev, 1.0))[0]

    def test_iterate_simple_normal_update(self, key):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, _w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            new_tr, _w, _rd, _bwd_request = jax.jit(scanner.update)(
                sub_key,
                tr,
                C[i, "z"].set(1.0),
                Diff.no_change((0.01,)),
            )
            assert new_tr.get_sample()[i, "z"] == 1.0


@genjax.gen
def inc(prev: ArrayLike) -> ArrayLike:
    return prev + 1


@genjax.gen
def inc_tupled(arg: tuple[ArrayLike, ArrayLike]) -> tuple[ArrayLike, ArrayLike]:
    """Takes a pair, returns a pair."""
    prev, offset = arg
    return (prev + offset, offset)


class TestIterate:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

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
        assert jnp.array_equal(jnp.asarray(result), jnp.array([0, 1, 2, 3, 4]))

        # same as result, with a jnp.array-wrapped accumulator
        result_wrapped = inc.iterate(n=4).simulate(key, (jnp.array(0),)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), result_wrapped)

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
        assert jnp.array_equal(jnp.asarray(result), jnp.array((2, 2)))

    def test_iterate_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation.
        """
        result = inc_tupled.iterate(n=4).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(
            jnp.asarray(result),
            jnp.array([[0, 2, 4, 6, 8], [2, 2, 2, 2, 2]]),
        )

    def test_iterate_final_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation. Same idea as above, but with
        `iterate_final`.
        """
        result = inc_tupled.iterate_final(n=10).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((20, 2)))

    def test_iterate_array(self, key):
        """
        `iterate` on function with an array-shaped initial value works correctly.
        """

        @genjax.gen
        def double(prev):
            return prev + prev

        result = double.iterate(n=4).simulate(key, (jnp.ones(4),)).get_retval()

        assert jnp.array_equal(
            result,
            jnp.array([
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [4, 4, 4, 4],
                [8, 8, 8, 8],
                [16, 16, 16, 16],
            ]),
        )

    def test_iterate_matrix(self, key):
        """
        `iterate` on function with matrix-shaped initial value works correctly.
        """

        fibonacci_matrix = jnp.array([[1, 1], [1, 0]])

        @genjax.gen
        def fibonacci_step(prev):
            return fibonacci_matrix @ prev

        iterated_fib = fibonacci_step.iterate(n=5)
        result = iterated_fib.simulate(key, (fibonacci_matrix,)).get_retval()

        # sequence of F^n fibonacci matrices
        expected = jnp.array([
            [[1, 1], [1, 0]],
            [[2, 1], [1, 1]],
            [[3, 2], [2, 1]],
            [[5, 3], [3, 2]],
            [[8, 5], [5, 3]],
            [[13, 8], [8, 5]],
        ])

        assert jnp.array_equal(result, expected)


@genjax.gen
def add(carry, x):
    return carry + x


@genjax.gen
def add_tupled(acc, x):
    """accumulator state is a pair."""
    carry, offset = acc
    return (carry + x + offset, offset)


class TestAccumulateReduceMethods:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_add(self, key):
        """Baseline test that `add` works!"""
        result = add.simulate(key, (0, 2)).get_retval()
        assert result == 2

    def test_accumulate(self, key):
        """
        `accumulate` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns an array of each intermediate accumulator value seen (not including the initial value).
        """
        result = add.accumulate().simulate(key, (0, jnp.ones(4))).get_retval()

        assert jnp.array_equal(result, jnp.array([0, 1, 2, 3, 4]))

        # same as result, but with a wrapped scalar vs a bare `0`.
        result_wrapped = (
            add.accumulate().simulate(key, (jnp.array(0), jnp.ones(4))).get_retval()
        )
        assert jnp.array_equal(result, result_wrapped)

    def test_reduce(self, key):
        """
        `reduce` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns the final `accumulator` produces by folding in each element of `jnp.array(v)`.
        """

        result = add.reduce().simulate(key, (0, jnp.ones(10))).get_retval()
        assert jnp.array_equal(result, 10)

    def test_add_tupled(self, key):
        """Baseline test demonstrating `add_tupled`."""
        result = add_tupled.simulate(key, ((0, 2), 10)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((12, 2)))

    def test_accumulate_tupled(self, key):
        """
        `accumulate` on function with tupled carry state works correctly.
        """
        result = (
            add_tupled.accumulate().simulate(key, ((0, 2), jnp.ones(4))).get_retval()
        )
        assert jnp.array_equal(
            jnp.asarray(result), jnp.array([[0, 3, 6, 9, 12], [2, 2, 2, 2, 2]])
        )
        jax.numpy.hstack

    def test_reduce_tupled(self, key):
        """
        `reduce` on function with tupled carry state works correctly.
        """
        result = add_tupled.reduce().simulate(key, ((0, 2), jnp.ones(10))).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((30, 2)))

    def test_accumulate_array(self, key):
        """
        `accumulate` with an array-shaped accumulator works correctly, including the initial value.
        """
        result = add.accumulate().simulate(key, (jnp.ones(4), jnp.eye(4))).get_retval()

        assert jnp.array_equal(
            result,
            jnp.array([
                [1, 1, 1, 1],
                [2, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 2, 1],
                [2, 2, 2, 2],
            ]),
        )

    def test_accumulate_matrix(self, key):
        """
        `accumulate` on function with matrix-shaped initial value works correctly.
        """

        fib = jnp.array([[1, 1], [1, 0]])
        repeated_fib = jnp.broadcast_to(fib, (5, 2, 2))

        @genjax.gen
        def matmul(prev, next):
            return prev @ next

        fib_steps = matmul.accumulate()
        result = fib_steps.simulate(key, (fib, repeated_fib)).get_retval()

        # sequence of F^n fibonacci matrices
        expected = jnp.array([
            [[1, 1], [1, 0]],
            [[2, 1], [1, 1]],
            [[3, 2], [2, 1]],
            [[5, 3], [3, 2]],
            [[8, 5], [5, 3]],
            [[13, 8], [8, 5]],
        ])

        assert jnp.array_equal(result, expected)


class TestScanUpdate:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_scan_update(self, key):
        @genjax.Pytree.dataclass
        class A(genjax.Pytree):
            x: FloatArray

        @genjax.gen
        def step(b, a):
            return genjax.normal(b + a.x, 1e-6) @ "b", None

        @genjax.gen
        def model(k):
            return step.scan(n=3)(k, A(jnp.array([1.0, 2.0, 3.0]))) @ "steps"

        k1, k2 = jax.random.split(key)
        tr = model.simulate(k1, (jnp.array(1.0),))
        u, w, _, _ = tr.update(k2, C["steps", 1, "b"].set(99.0))
        assert jnp.allclose(
            u.get_choices()["steps", ..., "b"], jnp.array([2.0, 99.0, 7.0]), atol=0.1
        )
        assert w < -100.0


class TestScanWithParameters:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    @genjax.gen
    @staticmethod
    def step(data, state, update):
        new_state = state + genjax.normal(update, data["noise"]) @ "state"
        return new_state, new_state

    @genjax.gen
    @staticmethod
    def model(data):
        stepper = TestScanWithParameters.step.partial_apply(data)
        return stepper.scan(n=3)(data["initial"], data["updates"]) @ "s"

    def test_scan_with_parameters(self, key):
        tr = TestScanWithParameters.model.simulate(
            key,
            (
                {
                    "initial": jnp.array(3.0),
                    "updates": jnp.array([5.0, 6.0, 7.0]),
                    "noise": 1e-6,
                },
            ),
        )

        end, steps = tr.get_retval()

        assert jnp.allclose(steps, jnp.array([8.0, 14.0, 21.0]), atol=0.1)
        assert jnp.allclose(end, jnp.array(21.0), atol=0.1)

    def test_scan_length_inferred(self, key):
        @genjax.gen
        def walk_step(x, std):
            new_x = genjax.normal(x, std) @ "x"
            return new_x, new_x

        args = (0.0, jnp.array([2.0, 4.0, 3.0, 5.0, 1.0]))
        tr = walk_step.scan(n=5).simulate(key, args)
        _, expected = tr.get_retval()
        assert jnp.allclose(
            tr.get_choices()[..., "x"],
            expected,
        )

        tr = walk_step.scan().simulate(key, args)
        assert jnp.allclose(tr.get_choices()[..., "x"], expected)

        # now with jit
        jitted = jax.jit(walk_step.scan().simulate)
        tr = jitted(key, args)
        assert jnp.allclose(tr.get_choices()[..., "x"], expected)

    def test_zero_length_scan(self, key):
        # GEN-333
        @genjax.gen
        def step(state, sigma):
            new_x = genjax.normal(state, sigma) @ "x"
            return (new_x, new_x + 1)

        trace = step.scan(n=0).simulate(key, (2.0, jnp.arange(0, dtype=float)))

        assert (
            trace.get_choices().static_is_empty()
        ), "zero-length scan produces empty choicemaps."

        key, subkey = jax.random.split(key)
        step.scan().importance(
            subkey,
            trace.get_choices(),
            (2.0, 2.0 + jnp.arange(0, dtype=float)),
        )

    def test_scan_validation(self, key):
        @genjax.gen
        def foo(shift, d):
            loc = d["loc"]
            scale = d["scale"]
            x = genjax.normal(loc, scale) @ "x"
            return x + shift, None

        d = {
            "loc": jnp.array([10.0, 12.0]),
            "scale": jnp.array([1.0]),
        }
        with pytest.raises(
            ValueError, match="scan got values with different leading axis sizes: 2, 1."
        ):
            jax.jit(foo.scan().simulate)(key, (jnp.array([1.0]), d))

    def test_vmap_key_scan(self, key):
        @genjax.gen
        def model(x, _):
            y = genjax.normal(x, 1.0) @ "y"
            return y, None

        vmapped = model.scan()

        keys = jax.random.split(key, 10)
        xs = jnp.arange(5, dtype=float)
        args = (jnp.array(1.0), xs)

        results = jax.vmap(lambda k: vmapped.simulate(k, args))(jnp.array(keys))

        chm = results.get_choices()

        # the inner scan aggregates a score, while the outer vmap does not accumulate anything
        assert results.get_score().shape == (10,)

        # the inner scan has scanned over the y's
        assert chm[..., "y"].shape == (10, 5)


class TestScanRegenerate:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_scan_regenerate(self, key):
        @genjax.gen
        def scanned_normal():
            @genjax.gen
            def kernel(carry, _):
                z = genjax.normal(0.0, 1.0) @ "z"
                return z, None

            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(0.0, 1.0) @ "y2"
            return kernel.scan(n=10)(y1, None) @ "kernel"

        key, sub_key = jax.random.split(key)
        tr = scanned_normal.simulate(sub_key, ())
        # First, try y1 and test for correctness.
        old_y1 = tr.get_choices()["y1"]
        old_target_density = genjax.normal.logpdf(old_y1, 0.0, 1.0)
        request = genjax.Regenerate(S.at["y1"])
        new_tr, fwd_w, _, _ = request.edit(key, tr, ())
        new_y1 = new_tr.get_choices()["y1"]
        new_target_density = genjax.normal.logpdf(new_y1, 0.0, 1.0)
        assert fwd_w == new_target_density - old_target_density


class TestScanIndexRequest:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_scan_regenerate(self):
        @genjax.gen
        def scanned_normal():
            @genjax.gen
            def kernel(carry, _):
                z = genjax.normal(0.0, 1.0) @ "z"
                return z, None

            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(0.0, 1.0) @ "y2"
            return kernel.scan(n=10)(y1, None) @ "kernel"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = scanned_normal.simulate(sub_key, ())
        # Try all indices and test for correctness.
        for idx in range(10):
            old_z = tr.get_choices()["kernel", idx, "z"]
            old_target_density = genjax.normal.logpdf(old_z, 0.0, 1.0)
            request = StaticRequest({
                "kernel": IndexRequest(jnp.array(idx), Regenerate(S.at["z"])),
            })
            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_z = new_tr.get_choices()["kernel", idx, "z"]
            new_target_density = genjax.normal.logpdf(new_z, 0.0, 1.0)
            assert fwd_w == new_target_density - old_target_density

        with pytest.raises(AssertionError):
            idx = 11
            old_z = tr.get_choices()["kernel", idx, "z"]
            old_target_density = genjax.normal.logpdf(old_z, 0.0, 1.0)
            request = StaticRequest({
                "kernel": IndexRequest(jnp.array(idx), Regenerate(S.at["z"])),
            })
            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_z = new_tr.get_choices()["kernel", idx, "z"]
            new_target_density = genjax.normal.logpdf(new_z, 0.0, 1.0)
            assert fwd_w == new_target_density - old_target_density

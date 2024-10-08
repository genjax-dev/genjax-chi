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

import jax.numpy as jnp

from genjax._src.core.interpreters.staging import FlagOp, multi_switch, tree_choose


class TestFlag:
    def test_basic_operation(self):
        true_flags = [
            True,
            jnp.array(True),
            jnp.array([True, True]),
        ]
        false_flags = [
            False,
            jnp.array(False),
            jnp.array([False, False]),
        ]
        for t in true_flags:
            assert jnp.all(t)
            assert not jnp.all(FlagOp.not_(t))
            for f in false_flags:
                assert not jnp.all(f)
                assert not jnp.all(FlagOp.and_(t, f))
                assert jnp.all(FlagOp.or_(t, f))
                assert jnp.all(FlagOp.xor_(t, f))
            for u in true_flags:
                assert jnp.all(FlagOp.and_(t, u))
                assert jnp.all(FlagOp.or_(t, u))
                assert not jnp.all(FlagOp.xor_(t, u))
        for f1 in false_flags:
            for f2 in false_flags:
                assert not jnp.all(FlagOp.xor_(f1, f2))

    def test_where(self):
        assert FlagOp.where(True, 3.0, 4.0) == 3
        assert FlagOp.where(False, 3.0, 4.0) == 4
        assert FlagOp.where(jnp.array(True), 3.0, 4.0) == 3
        assert FlagOp.where(jnp.array(False), 3.0, 4.0) == 4


class TestTreeChoose:
    def test_static_integer_index(self):
        result = tree_choose(1, [10, 20, 30])
        assert result == 20

    def test_jax_array_index(self):
        """
        Test that tree_choose works correctly with JAX array indices.
        This test ensures that when given a JAX array as an index,
        the function selects the correct value from the list.
        """
        result = tree_choose(jnp.array(2), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(30))

    def test_heterogeneous_types(self):
        """
        Test that tree_choose correctly handles heterogeneous types.
        It should attempt to cast compatible types (like bool to int)
        and use the dtype of the result for consistency.
        """
        result = tree_choose(2, [True, 2, False])
        assert result == 0
        assert jnp.asarray(result).dtype == jnp.int32

    def test_wrap_mode(self):
        """
        Test that tree_choose wraps around when the index is out of bounds.
        This should work for both jnp.array indices and concrete integer indices.
        """
        # first, the jnp.array index case:
        result = tree_choose(jnp.array(3), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(10))

        # then the concrete index case:
        concrete_result = tree_choose(3, [10, 20, 30])
        assert jnp.array_equal(result, concrete_result)


class TestMultiSwitch:
    def test_multi_switch(self):
        def branch_0(x):
            return {"result": x + 1, "extra": True}

        def branch_1(x, y):
            return {"result": x * y, "extra": [x, y]}

        def branch_2(x, y, z):
            return {
                "result": x + y + z,
                "extra": {"sum": x + y + z, "product": x * y * z},
            }

        branches = [branch_0, branch_1, branch_2]
        arg_tuples = [(5,), (3, 4), (1, 2, 3)]

        # Test with static index — the return value is the list of all possible shapes with only the selected one filled in.
        assert multi_switch(0, branches, arg_tuples) == [
            {"extra": True, "result": jnp.array(6, dtype=jnp.int32)},
            {
                "extra": [jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)],
                "result": jnp.array(0, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(0, dtype=jnp.int32),
                    "sum": jnp.array(0, dtype=jnp.int32),
                },
                "result": jnp.array(0, dtype=jnp.int32),
            },
        ]

        assert multi_switch(1, branches, arg_tuples) == [
            {"extra": False, "result": jnp.array(0, dtype=jnp.int32)},
            {
                "extra": [jnp.array(3, dtype=jnp.int32), jnp.array(4, dtype=jnp.int32)],
                "result": jnp.array(12, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(0, dtype=jnp.int32),
                    "sum": jnp.array(0, dtype=jnp.int32),
                },
                "result": jnp.array(0, dtype=jnp.int32),
            },
        ]

        assert multi_switch(2, branches, arg_tuples) == [
            {"extra": False, "result": jnp.array(0, dtype=jnp.int32)},
            {
                "extra": [jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)],
                "result": jnp.array(0, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(6, dtype=jnp.int32),
                    "sum": jnp.array(6, dtype=jnp.int32),
                },
                "result": jnp.array(6, dtype=jnp.int32),
            },
        ]

        # Test with dynamic index
        dynamic_index = jnp.array(1)
        assert multi_switch(dynamic_index, branches, arg_tuples) == multi_switch(
            1, branches, arg_tuples
        )

        # Test with out of bounds index (should clamp)
        assert multi_switch(10, branches, arg_tuples) == multi_switch(
            2, branches, arg_tuples
        )

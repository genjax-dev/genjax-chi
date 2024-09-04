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

import jax.numpy as jnp

from genjax._src.core.generative.functional_types import staged_choose


class TestStagedChoose:
    def test_static_integer_index(self):
        result = staged_choose(1, [10, 20, 30])
        assert result == 20

    def test_jax_array_index(self):
        """
        Test that staged_choose works correctly with JAX array indices.
        This test ensures that when given a JAX array as an index,
        the function selects the correct value from the list.
        """
        result = staged_choose(jnp.array(2), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(30))

    def test_heterogeneous_types(self):
        """
        Test that staged_choose correctly handles heterogeneous types.
        It should attempt to cast compatible types (like bool to int)
        and use the dtype of the result for consistency.
        """
        result = staged_choose(2, [True, 2, False])
        assert result == 0
        assert jnp.asarray(result).dtype == jnp.int32

    def test_wrap_mode(self):
        """
        Test that staged_choose wraps around when the index is out of bounds.
        This should work for both jnp.array indices and concrete integer indices.
        """
        # first, the jnp.array index case:
        result = staged_choose(jnp.array(3), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(10))

        # then the concrete index case:
        concrete_result = staged_choose(3, [10, 20, 30])
        assert jnp.array_equal(result, concrete_result)

# Copyright 2022 MIT Probabilistic Computing Project
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

from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey


# Convenient evolution to get multiple keys.
def slash(key: PRNGKey, num_pieces: IntArray):
    key, *sub_keys = jax.random.split(key, num_pieces + 1)
    sub_keys = jnp.array(sub_keys)
    return key, sub_keys

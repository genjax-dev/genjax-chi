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

import jax.tree_util as jtu
import numpy as np
import pyarrow as pa

from genjax.core.datatypes import Trace


def get_arrow_table(tr: Trace):
    values, forms = jtu.tree_flatten(tr)

    def _check_expand(v):
        if v.shape == ():
            return np.expand_dims(v, axis=-1)
        else:
            return v

    arrays = list(map(lambda v: pa.array(_check_expand(np.array(v))), values))
    table = pa.table(
        arrays, names=[str(ind) for (ind, _) in enumerate(values)]
    )
    return table

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


class TestSelections:
    def test_hierarchical_selection(self):
        s = genjax.select("x", ("z", "y"))
        assert s(("x",))
        assert s(("z",))
        assert s(("z", "y"))
        assert not s(("y",))
        assert not s(("z", "x"))

    def test_hierarchical_selection_filter(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(simple_normal.simulate)(key, ())
        selection = genjax.select("y1")
        chm = tr.filter(selection)
        assert chm["y1"] == tr["y1"]

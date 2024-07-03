
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
from genjax import ChoiceMapBuilder as C


class TestDimapCombinator:
    def test_dimap_update_retval(self):

        # Define pre- and post-processing functions
        def pre_process(x, y):
            return (x + 1, y * 2)

        def post_process(args, retval):
            return retval**2

        @genjax.gen
        def model(x, y):
            return genjax.normal(x, y) @ "z"

        dimap_model = model.dimap(
            pre=pre_process, post=post_process, info="Square of normal"
        )

        # Use the dimap model
        key = jax.random.PRNGKey(0)
        trace = dimap_model.simulate(key, (2.0, 3.0))  
        trace.get_retval()
        updated_tr, updated_w, _, _ = trace.update(key, C['z'].set(-2.0))
        assert 4.0 == updated_tr.get_retval(), "updating 'z' should run through `post_process` before returning"

        tr, w = dimap_model.importance(key, updated_tr.get_sample(), (2.0, 3.0))
        assert tr.get_retval() == updated_tr.get_retval()

        # TODO what can I say about the weight??
        # assert w == updated_w
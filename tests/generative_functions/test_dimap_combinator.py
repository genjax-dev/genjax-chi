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

import jax

import genjax
from genjax import ChoiceMapBuilder as C


class TestDimapCombinator:
    def test_dimap_update_retval(self):
        # Define pre- and post-processing functions
        def pre_process(x, y):
            return (x + 1, y * 2, y * 3)

        def post_process(_args, _xformed, retval):
            assert len(_args) == 2, "post_process receives pre-transformed args..."
            assert len(_xformed) == 3, "...and post-transformed args."
            return retval + 2

        def invert_post(x):
            return x - 2

        @genjax.gen
        def model(x, y, _):
            return genjax.normal(x, y) @ "z"

        dimap_model = model.dimap(pre=pre_process, post=post_process)

        # Use the dimap model
        key = jax.random.key(0)
        trace = dimap_model.simulate(key, (2.0, 3.0))
        assert trace.get_retval() == trace.get_choices()["z"] + 2.0, (
            "initial retval is a square of random draw"
        )

        assert (trace.get_score(), trace.get_retval()) == dimap_model.assess(
            trace.get_choices(), (2.0, 3.0)
        ), "assess with the same args returns score, retval"

        assert (
            genjax.normal.logpdf(
                invert_post(trace.get_retval()), *pre_process(2.0, 3.0)
            )
            == trace.get_score()
        ), (
            "final score sees pre-processing but not post-processing (note the inverse). This is only true here because we are returning the sampled value."
        )

        updated_tr, _, _, _ = trace.update(key, C["z"].set(-2.0))
        assert 0.0 == updated_tr.get_retval(), (
            "updated 'z' must hit `post_process` before returning"
        )

        importance_tr, _ = dimap_model.importance(
            key, updated_tr.get_choices(), (1.0, 2.0)
        )
        assert importance_tr.get_retval() == updated_tr.get_retval(), (
            "importance shouldn't update the retval"
        )

        assert (
            genjax.normal.logpdf(
                invert_post(importance_tr.get_retval()), *pre_process(1.0, 2.0)
            )
            == importance_tr.get_score()
        ), (
            "with importance trace, final score sees pre-processing but not post-processing."
        )

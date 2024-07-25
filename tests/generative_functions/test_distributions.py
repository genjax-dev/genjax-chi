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
from genjax import ChoiceMap as CM
from genjax import ChoiceMapBuilder as C
from genjax import ValueSample
from genjax.incremental import Diff, NoChange, UnknownChange


class TestDistributions:
    def test_simulate(self):
        key = jax.random.PRNGKey(314159)
        tr = genjax.normal.simulate(key, (0.0, 1.0))
        assert (
            tr.get_score() == genjax.normal.assess(key, tr.get_sample(), (0.0, 1.0))[0]
        )

    def test_importance(self):
        key = jax.random.PRNGKey(314159)

        # No constraint.
        (tr, w) = genjax.normal.importance(key, C.n(), (0.0, 1.0))
        assert w == 0.0

        # Constraint, no mask.
        (tr, w) = genjax.normal.importance(key, C.v(1.0), (0.0, 1.0))
        v = tr.get_sample()
        assert w == genjax.normal.assess(key, v, (0.0, 1.0))[0]

        # Constraint, mask with True flag.
        (tr, w) = genjax.normal.importance(
            key,
            CM.maybe(True, C.v(1.0)),
            (0.0, 1.0),
        )
        v = tr.get_sample()
        assert isinstance(v, ValueSample)
        assert v.val == 1.0
        assert w == genjax.normal.assess(key, v, (0.0, 1.0))[0]

        # Constraint, mask with False flag.
        (tr, w) = genjax.normal.importance(
            key,
            CM.maybe(False, C.v(1.0)),
            (0.0, 1.0),
        )
        v = tr.get_sample()
        assert isinstance(v, ValueSample)
        assert v.val != 1.0
        assert w == 0.0

    def test_update(self):
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = genjax.normal.simulate(sub_key, (0.0, 1.0))

        # No constraint, no change to arguments.
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.n(),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_sample().val == tr.get_sample().val
        assert (
            new_tr.get_score()
            == genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint, no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_sample().val == 1.0
        assert new_tr.get_score() == genjax.normal.assess(key, C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(key, C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

        # No constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.n(),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_sample() == tr.get_sample()
        assert (
            new_tr.get_score()
            == genjax.normal.assess(key, tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(key, tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0),
            (Diff(1.0, UnknownChange), Diff(2.0, UnknownChange)),
        )
        assert new_tr.get_sample().val == 1.0
        assert new_tr.get_score() == genjax.normal.assess(key, C.v(1.0), (1.0, 2.0))[0]
        assert (
            w
            == genjax.normal.assess(key, C.v(1.0), (1.0, 2.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            CM.maybe(True, C.v(1.0)),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_sample().val == 1.0
        assert new_tr.get_score() == genjax.normal.assess(key, C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(key, C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            CM.maybe(True, C.v(1.0)),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(key, C.v(1.0), (1.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(key, C.v(1.0), (1.0, 1.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (False), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            CM.maybe(False, C.v(1.0)),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score()
            == genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint is masked (False), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            CM.maybe(False, C.v(1.0)),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score()
            == genjax.normal.assess(key, tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(key, tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(key, tr.get_choices(), (0.0, 1.0))[0]
        )

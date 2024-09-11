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

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import EmptyConstraint, MaskedConstraint
from genjax import UpdateProblemBuilder as U
from genjax.incremental import Diff, NoChange, UnknownChange


class TestDistributions:
    def test_simulate(self):
        key = jax.random.PRNGKey(314159)
        tr = genjax.normal(0.0, 1.0).simulate(key, ())
        assert tr.get_score() == genjax.normal(0.0, 1.0).assess(tr.get_choices(), ())[0]

    def test_importance(self):
        key = jax.random.PRNGKey(314159)

        # No constraint.
        (tr, w) = genjax.normal.importance(key, EmptyConstraint(), (0.0, 1.0))
        assert w == 0.0

        # Constraint, no mask.
        (tr, w) = genjax.normal.importance(key, C.v(1.0), (0.0, 1.0))
        v = tr.get_choices()
        assert w == genjax.normal(0.0, 1.0).assess(v, ())[0]

        # Constraint, mask with True flag.
        (tr, w) = genjax.normal.importance(
            key,
            MaskedConstraint(True, C.v(1.0)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v == 1.0
        assert w == genjax.normal.assess(C.v(v), (0.0, 1.0))[0]

        # Constraint, mask with False flag.
        (tr, w) = genjax.normal.importance(
            key,
            MaskedConstraint(False, C.v(1.0)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v != 1.0
        assert w == 0.0

    def test_update(self):
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = genjax.normal.simulate(sub_key, (0.0, 1.0))

        # No constraint, no change to arguments.
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key, tr, U.g((Diff(0.0, NoChange), Diff(1.0, NoChange)), C.n())
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint, no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(0.0, NoChange), Diff(1.0, NoChange)),
                C.v(1.0),
            ),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # No constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g((Diff(1.0, UnknownChange), Diff(1.0, NoChange)), C.n()),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(1.0, UnknownChange), Diff(2.0, UnknownChange)),
                C.v(1.0),
            ),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (1.0, 2.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (1.0, 2.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(0.0, NoChange), Diff(1.0, NoChange)),
                MaskedConstraint(True, C.v(1.0)),
            ),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
                MaskedConstraint(True, C.v(1.0)),
            ),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (False), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(0.0, NoChange), Diff(1.0, NoChange)),
                MaskedConstraint(False, C.v(1.0)),
            ),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint is masked (False), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            U.g(
                (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
                MaskedConstraint(False, C.v(1.0)),
            ),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

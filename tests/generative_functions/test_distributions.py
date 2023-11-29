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
from genjax import ChoiceValue
from genjax import EmptyChoice
from genjax import mask
from genjax.incremental import NoChange
from genjax.incremental import UnknownChange
from genjax.incremental import diff


class TestDistributions:
    def test_simulate(self):
        key = jax.random.PRNGKey(314159)
        tr = genjax.normal.simulate(key, (0.0, 1.0))
        assert tr.get_score() == genjax.normal.logpdf(tr.get_value(), 0.0, 1.0)

    def test_importance(self):
        key = jax.random.PRNGKey(314159)

        # No constraint.
        (tr, w) = genjax.normal.importance(key, EmptyChoice(), (0.0, 1.0))
        assert w == 0.0

        # Constraint, no mask.
        (tr, w) = genjax.normal.importance(
            key,
            ChoiceValue(1.0),
            (0.0, 1.0),
        )
        v = tr.strip().get_value()
        assert w == genjax.normal.logpdf(v, 0.0, 1.0)

        # Constraint, mask with True flag.
        (tr, w) = genjax.normal.importance(
            key,
            mask(True, ChoiceValue(1.0)),
            (0.0, 1.0),
        )
        v = tr.strip().get_value()
        assert v == 1.0
        assert w == genjax.normal.logpdf(v, 0.0, 1.0)

        # Constraint, mask with False flag.
        (tr, w) = genjax.normal.importance(
            key,
            mask(False, ChoiceValue(1.0)),
            (0.0, 1.0),
        )
        v = tr.strip().get_value()
        assert v != 1.0
        assert w == 0.0

    def test_update(self):
        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = genjax.normal.simulate(sub_key, (0.0, 1.0))

        # No constraint, no change to arguments.
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            EmptyChoice(),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == tr.get_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_value(), 0.0, 1.0)
        assert w == 0.0

        # Constraint, no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            ChoiceValue(1.0),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 0.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 0.0, 1.0) - genjax.normal.logpdf(
            tr.get_value(), 0.0, 1.0
        )

        # No constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            EmptyChoice(),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == tr.get_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_value(), 1.0, 1.0)
        assert w == genjax.normal.logpdf(
            tr.get_value(), 1.0, 1.0
        ) - genjax.normal.logpdf(tr.get_value(), 0.0, 1.0)

        # Constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            ChoiceValue(1.0),
            (diff(1.0, UnknownChange), diff(2.0, UnknownChange)),
        )
        assert new_tr.get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 1.0, 2.0)
        assert w == genjax.normal.logpdf(1.0, 1.0, 2.0) - genjax.normal.logpdf(
            tr.get_value(), 0.0, 1.0
        )

        # Constraint is masked (True), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            mask(True, ChoiceValue(1.0)),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 0.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 0.0, 1.0) - genjax.normal.logpdf(
            tr.get_value(), 0.0, 1.0
        )

        # Constraint is masked (True), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            mask(True, ChoiceValue(1.0)),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 1.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 1.0, 1.0) - genjax.normal.logpdf(
            tr.get_value(), 0.0, 1.0
        )

        # Constraint is masked (False), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            mask(False, ChoiceValue(1.0)),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == tr.get_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_value(), 0.0, 1.0)
        assert w == 0.0

        # Constraint is masked (False), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            mask(False, ChoiceValue(1.0)),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_value() == tr.get_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_value(), 1.0, 1.0)
        assert w == genjax.normal.logpdf(
            tr.get_value(), 1.0, 1.0
        ) - genjax.normal.logpdf(tr.get_value(), 0.0, 1.0)

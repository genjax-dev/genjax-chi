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

import genjax
import jax
import pytest

from genjax import ValueChoiceMap, diff, NoChange, UnknownChange, EmptyChoiceMap, mask


class TestDistributions:
    def test_simulate(self):
        key = jax.random.PRNGKey(314159)
        key, tr = genjax.normal.simulate(key, (0.0, 1.0))
        assert tr.get_score() == genjax.normal.logpdf(tr.get_leaf_value(), 0.0, 1.0)

    def test_update(self):
        key = jax.random.PRNGKey(314159)
        key, tr = genjax.normal.simulate(key, (0.0, 1.0))

        # No constraint, no change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            EmptyChoiceMap(),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == tr.get_leaf_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_leaf_value(), 0.0, 1.0)
        assert w == 0.0

        # Constraint, no change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            ValueChoiceMap(1.0),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 0.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 0.0, 1.0) - genjax.normal.logpdf(
            tr.get_leaf_value(), 0.0, 1.0
        )

        # No constraint, change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            EmptyChoiceMap(),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == tr.get_leaf_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_leaf_value(), 1.0, 1.0)
        assert w == genjax.normal.logpdf(
            tr.get_leaf_value(), 1.0, 1.0
        ) - genjax.normal.logpdf(tr.get_leaf_value(), 0.0, 1.0)

        # Constraint, change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            ValueChoiceMap(1.0),
            (diff(1.0, UnknownChange), diff(2.0, UnknownChange)),
        )
        assert new_tr.get_leaf_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 1.0, 2.0)
        assert w == genjax.normal.logpdf(1.0, 1.0, 2.0) - genjax.normal.logpdf(
            tr.get_leaf_value(), 0.0, 1.0
        )

        # Constraint is masked (True), no change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            mask(True, ValueChoiceMap(1.0)),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 0.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 0.0, 1.0) - genjax.normal.logpdf(
            tr.get_leaf_value(), 0.0, 1.0
        )

        # Constraint is masked (True), change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            mask(True, ValueChoiceMap(1.0)),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == 1.0
        assert new_tr.get_score() == genjax.normal.logpdf(1.0, 1.0, 1.0)
        assert w == genjax.normal.logpdf(1.0, 1.0, 1.0) - genjax.normal.logpdf(
            tr.get_leaf_value(), 0.0, 1.0
        )

        # Constraint is masked (False), no change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            mask(False, ValueChoiceMap(1.0)),
            (diff(0.0, NoChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == tr.get_leaf_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_leaf_value(), 0.0, 1.0)
        assert w == 0.0

        # Constraint is masked (False), change to arguments.
        key, (_, w, new_tr, _) = genjax.normal.update(
            key,
            tr,
            mask(False, ValueChoiceMap(1.0)),
            (diff(1.0, UnknownChange), diff(1.0, NoChange)),
        )
        assert new_tr.get_leaf_value() == tr.get_leaf_value()
        assert new_tr.get_score() == genjax.normal.logpdf(tr.get_leaf_value(), 1.0, 1.0)
        assert w == genjax.normal.logpdf(
            tr.get_leaf_value(), 1.0, 1.0
        ) - genjax.normal.logpdf(tr.get_leaf_value(), 0.0, 1.0)

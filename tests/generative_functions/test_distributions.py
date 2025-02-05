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
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax.incremental import Diff, NoChange, UnknownChange


class TestDistributions:
    def test_simulate(self):
        key = jax.random.key(314159)
        tr = genjax.normal(0.0, 1.0).simulate(key, ())
        assert tr.get_score() == genjax.normal(0.0, 1.0).assess(tr.get_choices(), ())[0]

    def test_importance(self):
        key = jax.random.key(314159)

        # No constraint.
        (tr, w) = genjax.normal.importance(key, C.n(), (0.0, 1.0))
        assert w == 0.0

        # Constraint, no mask.
        (tr, w) = genjax.normal.importance(key, C.v(1.0), (0.0, 1.0))
        v = tr.get_choices()
        assert w == genjax.normal(0.0, 1.0).assess(v, ())[0]

        # Constraint, mask with True flag.
        (tr, w) = genjax.normal.importance(
            key,
            C.v(1.0).mask(jnp.array(True)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v == 1.0
        assert w == genjax.normal.assess(C.v(v), (0.0, 1.0))[0]

        # Constraint, mask with False flag.
        (tr, w) = genjax.normal.importance(
            key,
            C.v(1.0).mask(jnp.array(False)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v != 1.0
        assert w == 0.0

    def test_update(self):
        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = genjax.normal.simulate(sub_key, (0.0, 1.0))

        # No constraint, no change to arguments.
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.n(),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
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
            C.v(1.0),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
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
            C.n(),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
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
            C.v(1.0),
            (Diff(1.0, UnknownChange), Diff(2.0, UnknownChange)),
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
            C.v(1.0).mask(jnp.array(True)),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
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
            C.v(1.0).mask(True),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
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
            C.v(1.0).mask(False),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
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
            C.v(1.0).mask(False),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
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

    def test_using_primitive_distributions(self):
        @genjax.gen
        def model():
            _ = (
                genjax.bernoulli(
                    0.5,
                )
                @ "a"
            )
            _ = genjax.beta(1.0, 1.0) @ "b"
            _ = genjax.beta_binomial(1.0, 1.0, 1.0) @ "c"
            _ = genjax.beta_quotient(1.0, 1.0, 1.0, 1.0) @ "d"
            _ = genjax.binomial(1.0, 0.5) @ "e"
            _ = (
                genjax.cauchy(
                    0.0,
                    1.0,
                )
                @ "f"
            )
            _ = (
                genjax.categorical(
                    [0.5, 0.5],
                )
                @ "g"
            )
            _ = (
                genjax.chi(
                    1.0,
                )
                @ "h"
            )
            _ = (
                genjax.chi2(
                    1.0,
                )
                @ "i"
            )
            _ = (
                genjax.dirichlet(
                    [
                        1.0,
                        1.0,
                    ],
                )
                @ "j"
            )
            _ = (
                genjax.dirichlet_multinomial(
                    1.0,
                    [
                        1.0,
                        1.0,
                    ],
                )
                @ "k"
            )
            _ = genjax.double_sided_maxwell(1.0, 1.0) @ "l"
            _ = (
                genjax.exp_gamma(
                    1.0,
                    1.0,
                )
                @ "m"
            )
            _ = (
                genjax.exp_inverse_gamma(
                    1.0,
                    1.0,
                )
                @ "n"
            )
            _ = (
                genjax.exponential(
                    1.0,
                )
                @ "o"
            )
            _ = (
                genjax.flip(
                    0.5,
                )
                @ "p"
            )
            _ = (
                genjax.gamma(
                    1.0,
                    1.0,
                )
                @ "q"
            )
            _ = (
                genjax.geometric(
                    0.5,
                )
                @ "r"
            )
            _ = (
                genjax.gumbel(
                    0.0,
                    1.0,
                )
                @ "s"
            )
            _ = genjax.half_cauchy(1.0, 1.0) @ "t"
            _ = genjax.half_normal(1.0) @ "u"
            _ = genjax.half_student_t(1.0, 1.0, 1.0) @ "v"
            _ = (
                genjax.inverse_gamma(
                    1.0,
                    1.0,
                )
                @ "w"
            )
            _ = (
                genjax.kumaraswamy(
                    1.0,
                    1.0,
                )
                @ "x"
            )
            _ = (
                genjax.laplace(
                    0.0,
                    1.0,
                )
                @ "y"
            )
            _ = (
                genjax.lambert_w_normal(
                    1.0,
                    1.0,
                )
                @ "z"
            )
            _ = (
                genjax.log_normal(
                    0.0,
                    1.0,
                )
                @ "aa"
            )
            _ = (
                genjax.logit_normal(
                    0.0,
                    1.0,
                )
                @ "bb"
            )
            _ = (
                genjax.moyal(
                    0.0,
                    1.0,
                )
                @ "cc"
            )
            _ = (
                genjax.multinomial(
                    1.0,
                    [0.5, 0.5],
                )
                @ "dd"
            )
            _ = (
                genjax.mv_normal(
                    [0.0, 0.0],
                    [[1.0, 0.0], [0.0, 1.0]],
                )
                @ "ee"
            )
            _ = (
                genjax.mv_normal_diag(
                    jnp.array([1.0, 1.0]),
                )
                @ "ff"
            )
            _ = (
                genjax.negative_binomial(
                    1.0,
                    0.5,
                )
                @ "gg"
            )
            _ = (
                genjax.non_central_chi2(
                    1.0,
                    1.0,
                )
                @ "hh"
            )
            _ = (
                genjax.normal(
                    0.0,
                    1.0,
                )
                @ "ii"
            )
            _ = (
                genjax.poisson(
                    1.0,
                )
                @ "kk"
            )
            _ = (
                genjax.power_spherical(
                    jnp.array([
                        0.0,
                        0.0,
                    ]),
                    1.0,
                )
                @ "ll"
            )
            _ = (
                genjax.skellam(
                    1.0,
                    1.0,
                )
                @ "mm"
            )
            _ = genjax.student_t(1.0, 1.0, 1.0) @ "nn"
            _ = (
                genjax.truncated_cauchy(
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
                @ "oo"
            )
            _ = (
                genjax.truncated_normal(
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
                @ "pp"
            )
            _ = (
                genjax.uniform(
                    0.0,
                    1.0,
                )
                @ "qq"
            )
            _ = (
                genjax.von_mises(
                    0.0,
                    1.0,
                )
                @ "rr"
            )
            _ = (
                genjax.von_mises_fisher(
                    jnp.array([
                        0.0,
                        0.0,
                    ]),
                    1.0,
                )
                @ "ss"
            )
            _ = (
                genjax.weibull(
                    1.0,
                    1.0,
                )
                @ "tt"
            )
            _ = (
                genjax.zipf(
                    2.0,
                )
                @ "uu"
            )
            return None

        key = jax.random.key(314159)
        _ = model.simulate(key, ())

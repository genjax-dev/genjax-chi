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
import jax.numpy as jnp
import pytest
from genjax import ChoiceMap
from genjax.incremental import NoChange, UnknownChange, diff
from genjax.inference.translator import extending_trace_translator
from genjax.typing import typecheck


class TestExtendingTraceTranslator:
    def test_extending_trace_translator_vs_manual_update(self):
        @genjax.Unfold(max_length=10)
        @genjax.Static
        def model(z):
            z = genjax.normal(z, 1.0) @ "z"
            _x = genjax.normal(z, 1.0) @ "x"
            return z

        @genjax.Static
        @typecheck
        def proposal(obs_chm: ChoiceMap, prev_particle: ChoiceMap, *args):
            t = None
            masked_x = obs_chm[t, "x"]
            x = masked_x.unmask()
            z = genjax.normal(x, 0.01) @ "z"
            return z

        def get_translator(t, obs):
            @genjax.Static
            @typecheck
            def proposal(obs_chm: ChoiceMap, prev_particle: ChoiceMap, *args):
                masked_x = obs_chm[t, "x"]
                x = masked_x.unmask()
                z = genjax.normal(x, 0.01) @ "z"
                return z

            def choice_map_forward(proposal_choices):
                return genjax.indexed_choice_map(t, proposal_choices)

            def choice_map_inverse(transformed_choices):
                return transformed_choices[t].unmask()

            translator = extending_trace_translator(
                (diff(t, UnknownChange), diff(0.0, NoChange)),
                proposal,
                (),
                obs,
                choice_map_forward,
                choice_map_inverse,
                check_bijection=False,
            )
            return translator

        key = jax.random.PRNGKey(314159)

        # Starting trace.
        key, sub_key = jax.random.split(key)
        t1 = model.simulate(sub_key, (1, 0.0))

        # Observations, indexed for each time step.
        obs = genjax.indexed_choice_map(2, genjax.choice_map({"x": 3.0}))
        translator = get_translator(2, obs)
        key, sub_key = jax.random.split(key)
        t2, log_weight = translator(sub_key, t1)

        proposal_choices = genjax.choice_map({"z": t2[2, "z"]})
        proposal_weight = proposal.assess(proposal_choices, ())
        constraints = genjax.indexed_choice_map(
            2, genjax.choice_map({"x": 5.0, "z": t2[2, "z"]})
        )
        t3, up_weight = model.update(key, t1, constraints, (diff(2, UnknownChange),))
        assert log_weight == pytest.approx(up_weight - proposal_weight, 1e-4)


class TestExtendingTraceTranslator:
    def test_extending_trace_translator_vs_manual_update(self):
        @genjax.Unfold(max_length=10)
        @genjax.Static
        def model(z):
            z = genjax.normal(z, 1.0) @ "z"
            _x = genjax.normal(z, 1.0) @ "x"
            return z

        def get_translator(t, obs):
            @genjax.Static
            @typecheck
            def proposal(obs_chm: ChoiceMap, prev_particle: ChoiceMap, *args):
                masked_x = obs_chm[t, "x"]
                x = masked_x.unmask()
                z = genjax.normal(x, 0.01) @ "z"
                return z

            def choice_map_forward(proposal_choices):
                return genjax.indexed_choice_map(t, proposal_choices)

            def choice_map_inverse(transformed_choices):
                return transformed_choices[t].unmask()

            translator = extending_trace_translator(
                (diff(t, UnknownChange), diff(0.0, NoChange)),
                proposal,
                (),
                obs,
                choice_map_forward,
                choice_map_inverse,
                check_bijection=False,
            )
            return translator

        key = jax.random.PRNGKey(6)

        # Starting trace.
        key, sub_key = jax.random.split(key)
        t1 = model.simulate(sub_key, (1, 0.0))

        # Observations, indexed for each time step.
        obs = genjax.indexed_choice_map(jnp.array(2), genjax.choice_map({"x": 5.0}))
        translator = get_translator(2, obs)
        key, sub_key = jax.random.split(key)
        t2, log_weight = translator(sub_key, t1)

        # Proposal slice at 2.
        @genjax.Static
        @typecheck
        def proposal(obs_chm: ChoiceMap, *args):
            masked_x = obs_chm[2, "x"]
            x = masked_x.unmask()
            z = genjax.normal(x, 0.01) @ "z"
            return z

        proposal_choices = genjax.choice_map({"z": t2[2, "z"].unsafe_unmask()})
        proposal_weight, _ = proposal.assess(proposal_choices, (obs,))
        constraints = genjax.indexed_choice_map(
            jnp.array(2), genjax.choice_map({"x": 5.0, "z": t2[2, "z"].unsafe_unmask()})
        )
        t3, up_weight, *_ = model.update(
            key, t1, constraints, (diff(2, UnknownChange), diff(0.0, NoChange))
        )
        assert log_weight == pytest.approx(up_weight - proposal_weight, 1e-4)

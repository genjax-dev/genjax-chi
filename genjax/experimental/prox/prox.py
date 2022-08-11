# Copyright 2022 MIT ProbComp
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

from ...generative_function import simulate, generate
from typing import (
    Callable,
)


class ApproximateDensity:
    simulate: Callable
    estimate: Callable

    def __init__(self, simulate: Callable, estimate: Callable):
        self.simulate = simulate
        self.estimate = estimate


def complement(target):
    return lambda chm: filter(lambda x: not x[0] in target, chm.items())


def pseudomarginal(p, q, strategy, target):
    comp = complement(target)

    def sim(p_args, q_args, inf_args):
        tr = simulate(p)(p_args)
        model_score = tr.get_score()
        inf_sim, inf_est = strategy(p, q, *inf_args)
        w = inf_est(tr, p_args, q_args)
        return model_score - w, comp(tr.get_choices())

    def est(chm, p_args, q_args, inf_args):
        restricted = comp(chm)
        inf_sim, inf_est = strategy(p, q, *inf_args)
        tr = inf_sim(p_args, q_args)
        backwards = tr.get_score()
        forwards, _ = generate(p)(chm, *p_args)
        return forwards - backwards

    return ApproximateDensity(sim, est)


def importance(p, q, num_particles):
    return

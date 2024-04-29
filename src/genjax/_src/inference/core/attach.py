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

from genjax._src.core.generative.core import (
    Constraint,
    EmptySample,
    GenerativeFunction,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Callable, PRNGKey, Tuple, typecheck
from genjax._src.inference.core.kernel import KernelGenerativeFunction


class SMCMove(Pytree):
    pass


@Pytree.dataclass
class SMCP3Move(SMCMove):
    K: KernelGenerativeFunction
    L: KernelGenerativeFunction


@Pytree.dataclass
class DeferToInternal(SMCMove):
    pass


UpdateAttachment = Callable[[UpdateSpec], SMCMove]

ImportanceAttachment = Callable[[Constraint], SMCMove]


def default_attachment(_):
    return DeferToInternal()


@Pytree.dataclass
class AttachTrace(Trace):
    trace: Trace


@Pytree.dataclass
@typecheck
class AttachCombinator(GenerativeFunction):
    gen_fn: GenerativeFunction
    importance_attachment: ImportanceAttachment = Pytree.static(
        default=default_attachment,
    )
    update_attachment: UpdateAttachment = Pytree.static(
        default=default_attachment,
    )

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
    ) -> Trace:
        tr = self.gen_fn.simulate(key)
        return AttachTrace(tr)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        move = self.importance_attachment(constraint)
        match move:
            case SMCP3Move(K, L):
                K_tr = K.simulate(key)
                K_aux_score = K_tr.get_score()
                (new_latents, aux) = K_tr.get_retval()
                w_smc = K.smcp3_weight(
                    L,
                    K_aux_score,
                    EmptySample(),  # old latents
                    new_latents,  # new latents
                    aux,  # aux from K
                )
                tr, w, bwd = self.gen_fn.importance(key, new_latents)
                return tr, w + w_smc, bwd

            case DeferToInternal():
                return self.gen_fn.importance(key, constraint)

            case _:
                raise Exception("Invalid move type")

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: AttachTrace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        gen_fn_trace = trace.trace
        move = self.update_attachment(update_spec)
        match move:
            case SMCP3Move(K, L):
                K, L = self.custom_update_proposals(update_spec)
                K_tr = K.simulate(key)
                K_aux_score = K_tr.get_score()
                (new_latents, aux) = K_tr.get_retval()
                old_latents = trace.get_sample()
                w_smc = K.smcp3_weight(
                    L,
                    K_aux_score,
                    old_latents,  # old latents
                    new_latents,  # new latents
                    aux,  # aux from K
                )
                tr, w, retdiff, bwd_spec = self.gen_fn.update(
                    key, gen_fn_trace, new_latents
                )
                return tr, w + w_smc, retdiff, bwd_spec

            case DeferToInternal():
                return self.gen_fn.update(key, gen_fn_trace, update_spec)

            case _:
                raise Exception("Invalid move type")

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


import jax.numpy as jnp
import jax.random as jrand

from genjax._src.core.generative.choice_map import (
    ChoiceMap,
    Selection,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    EditRequest,
    PrimitiveEditRequest,
    Retdiff,
    Weight,
)
from genjax._src.core.generative.generative_function import (
    GenerativeFunction,
    Trace,
    Update,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
    TypeVar,
)

# Type variables
R = TypeVar("R")
ER = TypeVar("ER", bound=EditRequest)


@Pytree.dataclass(match_args=True)
class EmptyRequest(EditRequest):
    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        if Diff.static_check_no_change(argdiffs):
            return tr, jnp.array(0.0), Diff.no_change(tr.get_retval()), EmptyRequest()
        else:
            request = Update(ChoiceMap.empty())
            return request.edit(key, tr, argdiffs)


@Pytree.dataclass(match_args=True)
class Regenerate(PrimitiveEditRequest):
    selection: Selection


@Pytree.dataclass(match_args=True)
class Rejuvenate(EditRequest):
    """
    The `Rejuvenate` edit request is a compositional request which utilizes
    a proposal generative function to propose a change to a trace.

    Specifying a rejuvenation requires that a user provide a `proposal`
    generative function, and an `argument_mapping`,
    which is a callable that accepts the `ChoiceMap` from the previous trace
    and produces arguments to the invocation of the generative function.

    `Rejuvenate` can be used for quick custom regeneration moves (e.g. a Gaussian at an address,
    can I propose a random walk around the previous value using another Gaussian?)
    as well as larger structured proposals.

    The logic of this move is equivalent to the logic of
    Metropolis-Hastings with a custom proposal
    (which defines the MCMC kernel of the Metropolis-Hastings algorithm)
    _without the accept-reject step_. It uses the same proposal Q
    (provided in `Rejuvenate.proposal`) for the K and L ingredients of
    the SMCP3 move. The accept-reject ratio is returned as the SMCP3 weight of the move.
    """

    proposal: GenerativeFunction[Any]
    argument_mapping: Callable[[ChoiceMap], Any] = Pytree.static()

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[Any],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[Any], Weight, Retdiff[Any], "EditRequest"]:
        chm = tr.get_choices()
        fwd_proposal_args = self.argument_mapping(chm)
        key, sub_key = jrand.split(key)
        proposed_change, fwd_proposal_score, _ = self.proposal.propose(
            sub_key, fwd_proposal_args
        )
        request = Update(proposed_change)
        new_tr, w, retdiff, bwd_request = request.edit(key, tr, argdiffs)
        assert isinstance(bwd_request, Update)
        bwd_chm = bwd_request.constraint
        bwd_proposal_args = self.argument_mapping(bwd_chm)
        bwd_proposal_score, _ = self.proposal.assess(bwd_chm, bwd_proposal_args)
        final_weight = w + bwd_proposal_score - fwd_proposal_score
        return (
            new_tr,
            final_weight,
            retdiff,
            Rejuvenate(self.proposal, self.argument_mapping),
        )


# NOTE: can be used in an unsafe fashion!
@Pytree.dataclass(match_args=True)
class DiffAnnotate(Generic[ER], EditRequest):
    """
    The `DiffAnnotate` request can be used to introspect on the values of type `Diff` (primal and change tangent) values flowing
    through an edit program.

    Users can provide an `argdiff_fn` and a `retdiff_fn` to manipulate changes. Note that, this introspection is inherently unsafe, users should expect:

        * If you convert `Argdiffs` in such a way that you _assert_ that a value hasn't changed (when it actually has), the edit computation will be incorrect. Similar for the `Retdiff`.
    """

    request: ER
    argdiff_fn: Callable[[Argdiffs], Argdiffs] = Pytree.static(default=lambda v: v)
    retdiff_fn: Callable[[Retdiff[Any]], Retdiff[Any]] = Pytree.static(
        default=lambda v: v
    )

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        new_argdiffs = self.argdiff_fn(argdiffs)
        tr, w, retdiff, bwd_request = self.request.edit(key, tr, new_argdiffs)
        new_retdiff = self.retdiff_fn(retdiff)
        return tr, w, new_retdiff, bwd_request

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


from genjax._src.core.generative import (
    ChoiceMap,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    ImportanceProblem,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    PRNGKey,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class AddressBijectionTrace(Trace):
    gen_fn: "AddressBijectionCombinator"
    inner: Trace

    def get_args(self) -> Tuple:
        return self.inner.get_args()

    def get_retval(self) -> Any:
        return self.inner.get_retval()

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_sample(self) -> Sample:
        sample: ChoiceMap = self.inner.get_sample()
        return sample.with_addr_map(self.gen_fn.mapping)

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class AddressBijectionCombinator(GenerativeFunction):
    """
    Combinator that takes a [`genjax.GenerativeFunction`][] and a mapping from new addresses to old addresses and returns a new generative function with the same behavior but with the addresses transformed according to the mapping.

    Constraints passed into GFI methods on the returned [`genjax.GenerativeFunction`][] should use the new addresses (keys) and expect them to be mapped to the old addresses (values) internally. Any returned trace will have old addresses (values) mapped to new addresses (keys).

    !!! info
        Note that the `mapping` must be unique, or the constructor will throw an error.

    Attributes:
        gen_fn: The inner generative function to be transformed.
        mapping: A dictionary specifying the address mapping. Keys are original addresses, and values are the new addresses.

    Examples:
        Applying an address mapping to a generative function:
        ```python exec="yes" html="true" source="material-block" session="map_addresses"
        import genjax
        import jax


        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 1.0) @ "y"
            return x + y


        mapped_model = model.map_addresses(mapping={"new_x": "x", "new_y": "y"})

        key = jax.random.PRNGKey(0)
        trace = mapped_model.simulate(key, ())
        chm = trace.get_sample()
        print((chm["new_x"], chm["new_y"]))
        ```
    """

    gen_fn: GenerativeFunction
    mapping: dict = Pytree.static(default_factory=dict)

    def get_inverse(self) -> dict:
        inverse_map = {v: k for (k, v) in self.mapping.items()}
        return inverse_map

    def static_check_bijection(self):
        inverse_map = self.get_inverse()
        for k, v in self.mapping.items():
            assert inverse_map[v] == k

    def __post_init__(self):
        self.static_check_bijection()

    #################################
    # Generative function interface #
    #################################

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        tr = self.gen_fn.simulate(key, args)
        return AddressBijectionTrace(self, tr)

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ):
        inner_problem = chm.with_addr_map(self.get_inverse())
        tr, w, retdiff, inner_bwd_problem = self.gen_fn.update(
            key,
            EmptyTrace(self.gen_fn),
            GenericProblem(
                argdiffs,
                ImportanceProblem(inner_problem),
            ),
        )
        assert isinstance(inner_bwd_problem, ChoiceMap)
        bwd_problem = inner_bwd_problem.with_addr_map(self.mapping)
        return AddressBijectionTrace(self, tr), w, retdiff, bwd_problem

    @typecheck
    def update_choice_map(
        self,
        key: PRNGKey,
        trace: AddressBijectionTrace,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        inner_problem = chm.with_addr_map(self.get_inverse())
        tr, w, retdiff, inner_bwd_problem = self.gen_fn.update(
            key,
            trace.inner,
            GenericProblem(
                argdiffs,
                inner_problem,
            ),
        )
        assert isinstance(inner_bwd_problem, ChoiceMap)
        bwd_problem = inner_bwd_problem.with_addr_map(self.mapping)
        return AddressBijectionTrace(self, tr), w, retdiff, bwd_problem

    @typecheck
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case ChoiceMap():
                return self.update_choice_map(key, trace, update_problem, argdiffs)

            case ImportanceProblem(constraint):
                return self.update_importance(key, constraint, argdiffs)

            case _:
                raise ValueError(f"Unrecognized update problem: {update_problem}")

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_problem, Diff.no_change(trace.get_args())
                )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        match sample:
            case ChoiceMap():
                inner_sample = sample.with_addr_map(self.get_inverse())
                return self.gen_fn.assess(inner_sample, args)
            case _:
                raise ValueError(f"Not handled sample: {sample}")


@typecheck
def map_addresses(
    *,
    mapping: dict,
) -> Callable[[GenerativeFunction], AddressBijectionCombinator]:
    """
    Takes a mapping from new addresses to old addresses and returns a decorator that produces a [`genjax.GenerativeFunction`][] with the same behavior but with the addresses transformed according to the mapping.

    Constraints passed into GFI methods on the returned [`genjax.GenerativeFunction`][] should use the new addresses (keys) and expect them to be mapped to the old addresses (values) internally. Any returned trace will have old addresses (values) mapped to new addresses (keys).

    !!! info
        Note that the values in the `mapping` must be unique, or the constructor will throw an error.

    Args:
        mapping: A dictionary specifying the address mapping. Keys are original addresses, and values are the new addresses.

    Returns:
        A decorator that takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] with the same behavior but with the addresses transformed according to the mapping.

    Examples:
        Applying an address mapping to a generative function:
        ```python exec="yes" html="true" source="material-block" session="map_addresses"
        import genjax
        import jax


        @genjax.map_addresses(mapping={"new_x": "x", "new_y": "y"})
        @genjax.gen
        def mapped_model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 1.0) @ "y"
            return x + y


        key = jax.random.PRNGKey(0)
        trace = mapped_model.simulate(key, ())
        chm = trace.get_sample()
        print((chm["new_x"], chm["new_y"]))
        ```
    """

    def decorator(f):
        return AddressBijectionCombinator(f, mapping)

    return decorator

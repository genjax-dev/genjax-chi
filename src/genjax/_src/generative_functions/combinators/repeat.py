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

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
)
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import Callable, Int, IntArray, Tuple, typecheck
from genjax._src.generative_functions.static import gen
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Bool,
    BoolArray,
    Int,
    IntArray,
    Tuple,
    typecheck,
)

register_exclusion(__file__)

@Pytree.dataclass
class AddrMapChm(ChoiceMap):
    c: ChoiceMap = Pytree.field()
    mapping: dict = Pytree.static()

    @classmethod
    def create(cls, c: ChoiceMap, mapping: dict) -> ChoiceMap:
        if c.static_is_empty():
            return choice_map_empty
        return cls(c, mapping)

    def get_value(self) -> Bool | BoolArray:
        mapped = self.mapping.get((), ())
        if mapped:
            submap = self.c.get_submap(mapped)
            return submap.get_value()
        else:
            return self.c.get_value()

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        if ... in self.mapping:
            mapped = self.mapping[...]
            return self.c.get_submap(mapped).get_submap(addr)
        else:
            mapped = self.mapping.get(addr, addr)
            if mapped is ...:
                return self.c
            return self.c.get_submap(mapped)

@Pytree.dataclass
class MapAddressesTrace(Trace):
    gen_fn: "MapAddressesCombinator"
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
class MapAddressesCombinator(GenerativeFunction):
    """
    Combinator that takes a [`genjax.GenerativeFunction`][] and a mapping from new addresses to old addresses and returns a new generative function with the same behavior but with the addresses transformed according to the mapping.

    Constraints passed into GFI methods on the returned [`genjax.GenerativeFunction`][] should use the new addresses (keys) and expect them to be mapped to the old addresses (values) internally. Any returned trace will have old addresses (values) mapped to new addresses (keys).

    !!! info
        Note that the `mapping` must be unique, or the constructor will throw an error.

    Attributes:
        gen_fn: The inner generative function to be transformed.
        mapping: A dictionary specifying the address mapping. Keys are original addresses, and values are the new addresses.
    """

    gen_fn: GenerativeFunction
    mapping: dict = Pytree.static(default_factory=dict)

    @cached_property
    def inverse_mapping(self) -> dict:
        inverse_map = {v: k for (k, v) in self.mapping.items()}
        return inverse_map

    def _static_check_invertible(self):
        inverse_map = self.inverse_mapping
        for k, v in self.mapping.items():
            assert inverse_map[v] == k

    def __post_init__(self):
        self._static_check_invertible()

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
        return MapAddressesTrace(self, tr)

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ):
        inner_problem = chm.with_addr_map(self.inverse_mapping)
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
        return MapAddressesTrace(self, tr), w, retdiff, bwd_problem

    @typecheck
    def update_choice_map(
        self,
        key: PRNGKey,
        trace: MapAddressesTrace,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        inner_problem = chm.with_addr_map(self.inverse_mapping)
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
        return MapAddressesTrace(self, tr), w, retdiff, bwd_problem

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
                inner_sample = sample.with_addr_map(self.inverse_mapping)
                return self.gen_fn.assess(inner_sample, args)
            case _:
                raise ValueError(f"Not handled sample: {sample}")

def RepeatCombinator(gen_fn: GenerativeFunction, /, *, n: Int) -> GenerativeFunction:
    """
    A combinator that samples from a supplied [`genjax.GenerativeFunction`][] `gen_fn` a fixed number of times, returning a vector of `n` results.

    See [`genjax.repeat`][] for more details.
    """

    def argument_mapping(*args):
        return (jnp.zeros(n), args)

    # This is a static generative function with an attached
    # choicemap address mapping, to collapse the `_internal`
    # address hierarchy below.
    # (as part of StaticGenerativeFunction.Trace interfaces)
    @map_addresses(mapping={...: "_internal"})
    @gen
    def expanded_gen_fn(_: IntArray, args: Tuple):
        return gen_fn(*args) @ "_internal"

    return expanded_gen_fn.vmap(in_axes=(0, None)).contramap(
        argument_mapping, info="Derived combinator (Repeat)"
    )


@typecheck
def repeat(*, n: Int) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] `gen_fn` of type `a -> b` and returns a new `GenerativeFunction` of type `a -> [b]` that samples from `gen_fn `n` times, returning a vector of `n` results.

    The values traced by each call `gen_fn` will be nested under an integer index that matches the loop iteration index that generated it.

    This combinator is useful for creating multiple samples from the same generative model in a batched manner.

    Args:
        n: The number of times to sample from the generative function.

    Returns:
        A new [`genjax.GenerativeFunction`][] that samples from the original function `n` times.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="repeat"
        import genjax, jax


        @genjax.repeat(n=10)
        @genjax.gen
        def normal_draws(mean):
            return genjax.normal(mean, 1.0) @ "x"


        key = jax.random.PRNGKey(314159)

        # Generate 10 draws from a normal distribution with mean 2.0
        tr = jax.jit(normal_draws.simulate)(key, (2.0,))
        print(tr.render_html())
        ```
    """

    def decorator(gen_fn) -> GenerativeFunction:
        return RepeatCombinator(gen_fn, n=n)

    return decorator

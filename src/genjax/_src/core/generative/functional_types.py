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
from jax.experimental import checkify
from jax.tree_util import tree_map

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import (
    Flag,
    flag,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Generic,
    Int,
    TypeVar,
    static_check_is_concrete,
    typecheck,
)

R = TypeVar("R")

#########################
# Masking and sum types #
#########################


@Pytree.dataclass(match_args=True)
class Mask(Generic[R], Pytree):
    """The `Mask` datatype wraps a value in a Boolean flag which denotes whether the data is valid or invalid to use in inference computations.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `Sample` instances, and participate in generative and inference computations (like scores, and importance weights or density ratios). Invalid data **should** be considered unusable, and should be handled with care.

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    ## Encountering `Mask` in your computation

    When users see `Mask` in their computations, they are expected to interact with them by either:

    * Unmasking them using the `Mask.unmask` interface, a potentially unsafe operation.

    * Destructuring them manually, and handling the cases.

    ## Usage of invalid data

    If you use invalid `Mask(False, data)` data in inference computations, you may encounter silently incorrect results.
    """

    flag: Flag
    value: R

    @classmethod
    def maybe(cls, f: Flag, v: Any):
        match v:
            case Mask(flag, value):
                return Mask.maybe_none(f.and_(flag), value)
            case _:
                return Mask(f, v)

    @classmethod
    def maybe_none(cls, f: Flag, v: Any):
        return (
            None
            if v is None
            else v
            if f.concrete_true()
            else None
            if f.concrete_false()
            else Mask.maybe(f, v)
        )

    ######################
    # Masking interfaces #
    ######################

    def unmask(self) -> R:
        """Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime.
        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            checkify.check(
                self.flag.f,
                "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
            )

        optional_check(_check)
        return self.value

    def unsafe_unmask(self) -> R:
        # Unsafe version of unmask -- should only be used internally,
        # or carefully.
        return self.value


@Pytree.dataclass(match_args=True)
class Sum(Pytree):
    """
    A `Sum` instance represents a sum type, which is a union of possible values - which value is active is determined by the `Sum.idx` field.

    The `Sum` type is used to represent a choice between multiple possible values, and is used in generative computations to represent uncertainty over values.

    Examples:
        A common scenario which will produce `Sum` types is when using a `SwitchCombinator` with branches that have
        multiple possible return value types:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import gen, normal, bernoulli


        @gen
        def model1():
            return normal(0.0, 1.0) @ "x"


        @gen
        def model2():
            z = bernoulli(0.5) @ "z"
            return (z, z)


        tr = jax.jit(model1.switch(model2).simulate)(key, (1, (), ()))
        print(tr.get_retval().render_html())
        ```

        Users can collapse the `Sum` type by consuming it via [`jax.lax.switch`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html), for instance:
        ```python exec="yes" html="true" source="material-block" session="core"
        def collapsing_a_sum_type(key, idx):
            tr = model1.switch(model2).simulate(key, (idx, (), ()))
            sum = tr.get_retval()
            v = jax.lax.switch(
                sum.idx,
                [
                    lambda: sum.values[0] + 3.0,
                    lambda: 1.0 + sum.values[1][0] + sum.values[1][1],
                ],
            )
            return v


        x = jax.jit(collapsing_a_sum_type)(key, 1)
        print(x)
        ```

        Users can index into the `Sum` type using a **static** integer index, creating a `Mask` type:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import Sum


        def uncertain_idx(idx):
            s = Sum(idx, [1, 2, 3])
            return s[2]


        mask = jax.jit(uncertain_idx)(1)
        print(mask.render_html())
        ```
    """

    idx: ArrayLike | Diff
    """
    The runtime index tag for which value in `Sum.values` is active.
    """
    values: list[Any]
    """
    The possible values for the `Sum` instance.
    """

    @classmethod
    @typecheck
    def maybe(
        cls,
        idx: ArrayLike | Diff,
        vs: list[Any],
    ):
        return (
            vs[idx]
            if static_check_is_concrete(idx) and isinstance(idx, Int)
            else Sum(idx, list(vs)).maybe_collapse()
        )

    @classmethod
    @typecheck
    def maybe_none(
        cls,
        idx: ArrayLike | Diff,
        vs: list[Any],
    ):
        possibles = []
        for _idx, v in enumerate(vs):
            if v is not None:
                possibles.append(Mask.maybe_none(flag(idx == _idx), v))
        if not possibles:
            return None
        if len(possibles) == 1:
            return possibles[0]
        else:
            return Sum.maybe(idx, vs)

    def maybe_collapse(self):
        if Pytree.static_check_tree_structure_equivalence(self.values):
            idx = Diff.tree_primal(self.idx)
            return tree_map(lambda *vs: jnp.choose(idx, vs, mode="wrap"), *self.values)
        else:
            return self

    @typecheck
    def __getitem__(self, idx: Int):
        return Mask.maybe_none(flag(idx == self.idx), self.values[idx])

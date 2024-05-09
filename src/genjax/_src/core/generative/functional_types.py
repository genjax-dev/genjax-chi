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

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jax.tree_util import tree_map

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import (
    staged_and,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Bool,
    BoolArray,
    Callable,
    Int,
    IntArray,
    List,
    static_check_is_concrete,
    typecheck,
)

#########################
# Masking and sum types #
#########################


@Pytree.dataclass(match_args=True)
class Mask(Pytree):
    """The `Mask` datatype wraps a value in a Boolean flag which denotes whether the data is valid or invalid to use in inference computations. The masking system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `Sample` instances, and participate in generative and inference computations (like scores, and importance weights or density ratios). Invalid data **should** be considered unusable, and should be handled with care.

    !!! warning "Usage of invalid data"

        **If you use invalid `Mask(False, data)` data in inference computations, you may encounter silently incorrect results.**

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    ## Encountering `Mask` in your computation

    When users see `Mask` in their computations, they are expected to interact with them by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error if the `Mask.flag` value is invalid.

    * Destructuring them manually, and handling the cases.
    """

    flag: BoolArray
    value: Any

    @classmethod
    def maybe(cls, f: BoolArray, v: Any):
        match v:
            case Mask(flag, value):
                return Mask(staged_and(f, flag), value)
            case _:
                return (
                    v
                    if static_check_is_concrete(f) and isinstance(f, Bool) and f
                    else Mask(f, v)
                )

    @classmethod
    def maybe_none(cls, f: BoolArray, v: Any):
        if v is None or (static_check_is_concrete(f) and isinstance(f, Bool) and not f):
            return None
        return Mask.maybe(f, v)

    ######################
    # Masking interfaces #
    ######################

    def unmask(self):
        """Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime. To enforce validity checks, use the console context `genjax.console(enforce_checkify=True)` to handle any code which utilizes `Mask.unmask` with [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html).

        Examples:
            To enable runtime checks, the user must enable them explicitly in `genjax`.

        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.flag)
            checkify.check(
                check_flag,
                "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
            )

        optional_check(_check)
        return self.value

    def unsafe_unmask(self):
        # Unsafe version of unmask -- should only be used internally,
        # or carefully.
        return self.value

    @typecheck
    def match(self, some: Callable) -> Any:
        v = self.unmask()
        return some(v)

    @typecheck
    def safe_match(self, none: Callable, some: Callable) -> Any:
        return jax.lax.cond(
            self.flag,
            lambda v: some(v),
            lambda v: none(),
            self.value,
        )


@Pytree.dataclass(match_args=True)
class Sum(Pytree):
    idx: IntArray | Diff
    values: List

    @classmethod
    @typecheck
    def maybe(
        cls,
        idx: Int | IntArray | Diff,
        vs: List[ArrayLike | Pytree],
    ):
        return (
            vs[idx]
            if static_check_is_concrete(idx) and isinstance(idx, Int)
            else Sum(idx, list(vs))
        )

    @classmethod
    @typecheck
    def maybe_none(
        cls,
        idx: Int | IntArray | Diff,
        vs: List[None | ArrayLike | Pytree],
    ):
        possibles = []
        for _idx, v in enumerate(vs):
            if v is not None:
                possibles.append(Mask.maybe(idx == _idx, v))
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

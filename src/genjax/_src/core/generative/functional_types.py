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

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.staging import (
    staged_and,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
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


class Mask(Pytree):
    """The `Mask` datatype provides access to the masking system. The masking
    system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `Sample` instances, and participate in inference computations (like scores, and importance weights or density ratios).

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Users are expected to interact with `Mask` instances by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error if the `Mask.flag` value is invalid.

    * Using `Mask.match` - which allows a user to provide "none" and "some" lambdas. The "none" lambda should accept no arguments, while the "some" lambda should accept an argument whose type is the same as the masked value. These lambdas should return the same type (`Pytree`, array, etc) of value.
    """

    flag: BoolArray
    value: Any

    # If the user provides a `Mask` as the value, we merge the flags and unwrap
    # one layer of the structure.
    def __post_init__(self):
        if isinstance(self.value, Mask):
            self.flag = staged_and(self.flag, self.value.flag)
            self.value = self.value.value

    @classmethod
    def maybe(cls, f: BoolArray, v: Any):
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
        """> Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime. To enforce validity checks, use the console context `genjax.console(enforce_checkify=True)` to handle any code which utilizes `Mask.unmask` with [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax

            console = genjax.console()

            masked = genjax.Mask(True, jnp.ones(5))
            print(console.render(masked.unmask()))
            ```

            To enable runtime checks, the user must enable them explicitly in `genjax`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.experimental.checkify as checkify
            import jax.numpy as jnp
            import genjax

            with genjax.console(enforce_checkify=True) as console:
                masked = genjax.Mask(False, jnp.ones(5))
                err, _ = checkify.checkify(masked.unmask)()
                print(console.render(err))
            ```
        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.flag)
            checkify.check(
                check_flag,
                "Attempted to unmask when the mask flag is False: the masked value is invalid.\n",
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


class Sum(Pytree):
    idx: IntArray
    values: List[Any]

    @classmethod
    def maybe(cls, idx: BoolArray, v1: Any, v2: Any):
        return (
            [v1, v2][idx]
            if static_check_is_concrete(idx) and isinstance(idx, Int)
            else Sum(idx, [v1, v2])
        )

    @classmethod
    def maybe_none_or_mask(cls, idx: BoolArray, v1: Any, v2: Any):
        if v1 is None and v2 is None:
            return None
        elif v1 is None:
            return Mask.maybe_none(idx == 1, v2)
        elif v2 is None:
            return Mask.maybe_none(idx == 0, v1)
        return Sum.maybe(idx, v1, v2)

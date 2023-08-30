# Copyright 2022 MIT Probabilistic Computing Project
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

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.global_options import global_options


@dataclass
class Mask(Pytree):
    """The `Mask` datatype provides access to the masking system. The masking
    system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as constraints in choice maps, and participate in inference computations (like scores, and importance weights or density ratios).

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Users are expected to interact with `Mask` instances by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error.

    * Using `Mask.match` - which allows a user to provide "none" and "some" lambdas. The "none" lambda should accept no arguments, while the "some" lambda should accept an argument whose type is the same as the masked value. These lambdas should return the same type (`Pytree`, array, etc) of value.
    """

    mask: BoolArray
    value: Any

    def flatten(self):
        return (self.mask, self.value), ()

    @classmethod
    def new(cls, mask: BoolArray, inner):
        if isinstance(inner, Mask):
            return Mask(
                jnp.logical_and(mask, inner.mask),
                inner.value,
            )
        else:
            return cls(mask, inner)

    @typecheck
    def match(self, none: Callable, some: Callable) -> Any:
        """> Pattern match on the `Mask` type - by providing "none"
        and "some" lambdas.

        The "none" lambda should accept no arguments, while the "some" lambda should accept the same type as the value in the `Mask`. Both lambdas should return the same type (array, or `jax.Pytree`).

        Arguments:
            none: A lambda to handle the "none" branch. The type of the return value must agree with the "some" branch.
            some: A lambda to handle the "some" branch. The type of the return value must agree with the "none" branch.

        Returns:
            value: A value computed by either the "none" or "some" lambda, depending on if the `Mask` is valid (e.g. `Mask.mask` is `True`).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax
            console = genjax.pretty()

            masked = genjax.mask(False, jnp.ones(5))
            v1 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            masked = genjax.mask(True, jnp.ones(5))
            v2 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            print(console.render((v1, v2)))
            ```
        """

        return jax.lax.cond(
            self.mask,
            lambda: some(self.value),
            lambda: none(),
        )

    def unmask(self):
        """> Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` is valid at runtime. To enforce validity checks, use `genjax.global_options.allow_checkify(True)` and then handle any code which utilizes `Mask.unmask` with [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax
            console = genjax.pretty()

            masked = genjax.mask(True, jnp.ones(5))
            print(console.render(masked.unmask()))
            ```

            Here's an example which uses `jax.experimental.checkify`. To enable runtime checks, the user must enable them explicitly in `genjax`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import jax.experimental.checkify as checkify
            import genjax
            console = genjax.pretty()
            genjax.global_options.allow_checkify(True)

            masked = genjax.mask(False, jnp.ones(5))
            err, _ = checkify.checkify(masked.unmask)()
            print(console.render(err))

            genjax.global_options.allow_checkify(False)
            ```
        """
        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.mask)
            checkify.check(check_flag, "Mask is False, the masked value is invalid.\n")

        global_options.optional_check(_check)
        return self.value

    def unsafe_unmask(self):
        # Unsafe version of unmask -- should only be used internally.
        return self.value

    ###########
    # Dunders #
    ###########

    def __getattr__(self, name):
        sub = getattr(self.value, name)
        if isinstance(sub, Callable):

            @functools.wraps(sub)
            def wrapper(*args):
                v = sub(*args)
                return Mask(self.mask, v)

            return wrapper
        else:
            return Mask(self.mask, sub)

    def __getitem__(self, name):
        s = self.value[name]
        return Mask(self.mask, s)

    @dispatch
    def __eq__(self, other: "Mask"):
        return jnp.logical_and(
            jnp.logical_and(self.mask, other.mask),
            self.value == other.value,
        )

    @dispatch
    def __eq__(self, other: Any):
        return jnp.logical_and(
            self.mask,
            self.value == other,
        )

    def __hash__(self):
        hash1 = hash(self.value)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.mask, short_arrays=True)
        sub_tree = Tree(f"[bold](Mask, {doc})")
        if isinstance(self.value, Pytree):
            _ = self.value.__rich_tree__(sub_tree)
        else:
            val_tree = gpp.tree_pformat(self.value, short_arrays=True)
            sub_tree.add(val_tree)
        tree.add(sub_tree)
        return tree

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


##############
# Shorthands #
##############

mask = Mask.new

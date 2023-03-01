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

import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

from genjax._src.core.interpreters import context
from genjax._src.core.interpreters import primitives as prim
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FrozenSet
from genjax._src.core.typing import Hashable
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import Optional
from genjax._src.core.typing import String
from genjax._src.core.typing import Union


Value = Any

sow_p = jc.Primitive("sow")
sow_p.multiple_results = True


class SowEffect(effects.Effect):
    __repr__ = lambda _: "Sow"


sow_effect = SowEffect()

effects.remat_allowed_effects.add_type(SowEffect)
effects.control_flow_allowed_effects.add_type(SowEffect)
effects.lowerable_effects.add_type(SowEffect)


@sow_p.def_impl
def _sow_impl(*args, **_):
    return args


@sow_p.def_effectful_abstract_eval
def _sow_abstract_eval(*avals, **_):
    return avals, {sow_effect}


def _sow_jvp(primals, tangents, **kwargs):
    out_primals = sow_p.bind(*primals, **kwargs)
    return out_primals, tangents


ad.primitive_jvps[sow_p] = _sow_jvp


def _sow_transpose(cts_in, *args, **kwargs):
    del args, kwargs
    return cts_in


ad.primitive_transposes[sow_p] = _sow_transpose


def _sow_batch_rule(batched_args, batch_dims, **params):
    outs = sow_p.bind(*batched_args, **params)
    return outs, batch_dims


batching.primitive_batchers[sow_p] = _sow_batch_rule
mlir.register_lowering(sow_p, lambda c, *args, **kw: args)


def sow(value, *, tag: Hashable, name: String, mode: String = "strict", key=None):
    """Marks a value with a name and a tag.

    Args:
      value: A JAX value to be tagged and named.
      tag: a string representing the tag of the sown value.
      name: a string representing the name to sow the value with.
      mode: The mode by which to sow the value. There are three options: 1.
        `'strict'` - if another value is sown with the same name and tag in the
        same context, harvest will throw an error. 2. `'clobber'` - if another is
        value is sown with the same name and tag, it will replace this value 3.
        `'append'` - sown values of the same name and tag are appended to a
        growing list. Append mode assumes some ordering on the values being sown
        defined by data-dependence.
      key: an optional JAX value that will be tied into the sown value.

    Returns:
      The original `value` that was passed in.
    """
    value = jtu.tree_map(jc.raise_as_much_as_possible, value)
    if key is not None:
        value = prim.tie_in(key, value)
    flat_args, in_tree = jtu.tree_flatten(value)
    out_flat = sow_p.bind(*flat_args, name=name, tag=tag, mode=mode, tree=in_tree)
    return jtu.tree_unflatten(in_tree, out_flat)


@dataclasses.dataclass(frozen=True)
class HarvestSettings:
    """Contains the settings for a HarvestTrace."""

    tag: Hashable
    blocklist: FrozenSet[String]
    allowlist: Union[FrozenSet[String], None]
    exclusive: bool


@dataclasses.dataclass
class HarvestContext(context.Context):
    def get_custom_rule(self, primitive):
        return None

    def can_process(self, primitive):
        return primitive in [sow_p]

    def process_primitive(self, primitive, *args, **kwargs):
        if primitive is sow_p:
            return self.process_sow(*args, **kwargs)
        else:
            raise NotImplementedError

    def process_sow(self, *values, name, tag, mode, tree):
        """Handles a `sow` primitive in a `HarvestTrace`."""
        if mode not in {"strict", "append", "clobber"}:
            raise ValueError(f"Invalid mode: {mode}")
        if tag != self.settings.tag:
            if self.settings.exclusive:
                return values
            return sow_p.bind(*values, name=name, tag=tag, tree=tree, mode=mode)
        if self.settings.allowlist is not None and name not in self.settings.allowlist:
            return values
        if name in self.settings.blocklist:
            return values
        return self.handle_sow(*values, name=name, tag=tag, tree=tree, mode=mode)

    def handle_sow(self, *values, name, tag, mode, tree):
        raise NotImplementedError


@dataclasses.dataclass
class Reap(Pytree):
    value: Any
    metadata: Dict[String, Any]

    def flatten(self):
        return (self.value,), (self.metadata,)


@dataclasses.dataclass
class ReapContext(HarvestContext):
    settings: HarvestSettings
    reaps: Dict[String, Any]

    def flatten(self):
        return (self.settings, self.reaps), ()

    @classmethod
    def new(cls, settings):
        reaps = dict()
        return ReapContext(settings, reaps)

    def yield_state(self):
        return (self.reaps,)

    def handle_sow(self, *values, name, tag, tree, mode):
        """Stores a sow in the reaps dictionary."""
        del tag
        if name in self.reaps:
            raise ValueError(f"Variable has already been reaped: {name}")
        # TODO: revisit.
        # avals = jtu.tree_unflatten(
        #    tree,
        #    [abstract_arrays.raise_to_shaped(jc.get_aval(v)) for v in values],
        # )
        self.reaps[name] = jtu.tree_unflatten(tree, values)
        return values


def reap(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    blocklist = frozenset(blocklist)
    if allowlist is not None:
        allowlist = frozenset(allowlist)
    settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = ReapContext.new(settings)
        retvals, (reaps,) = context.transform(fn, ctx)(*args, **kwargs)
        return retvals, reaps

    return wrapper


plant_custom_rules = {}


@dataclasses.dataclass
class PlantContext(HarvestContext):
    """Contains the settings and storage for the current trace in the stack."""

    settings: HarvestSettings
    plants: Dict[str, Any]

    def flatten(self):
        return (self.plants,), (self.settings,)

    def __post_init__(self):
        self._already_planted = set()

    def yield_state(self):
        return ()

    def handle_sow(self, *values, name, tag, tree, mode):
        """Returns the value stored in the plants dictionary."""
        if name in self._already_planted:
            raise ValueError(f"Variable has already been planted: {name}")
        if name in self.plants:
            self._already_planted.add(name)
            return jtu.tree_leaves(self.plants[name])
        return sow_p.bind(*values, name=name, tag=tag, mode=mode, tree=tree)


def plant(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    blocklist = frozenset(blocklist)
    if allowlist is not None:
        allowlist = frozenset(allowlist)
    settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        ctx = PlantContext.new(settings, plants)
        retvals, _ = context.transform(fn, ctx)(*args, **kwargs)
        return retvals

    return wrapper


def harvest(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        f = plant(
            fn, tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive
        )
        f = reap(
            f, tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive
        )
        return f(plants, *args, **kwargs)

    return wrapper

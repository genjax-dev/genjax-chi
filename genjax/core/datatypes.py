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

import abc
import jax
import jax.tree_util as jtu
import numpy as np
from dataclasses import dataclass
from genjax.core.pytree import Pytree, squeeze
from genjax.core.tracetypes import Bottom
from typing import Any, Sequence, Union
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

#####
# GenerativeFunction
#####


@dataclass
class GenerativeFunction(Pytree):
    """
    :code:`GenerativeFunction` abstract class which allows user-defined
    implementations of the generative function interface methods.
    The :code:`builtin` and :code:`distributions` languages both
    implement a class inheritor of :code:`GenerativeFunction`.

    Any implementation will interact with the JAX tracing machinery,
    however, so there are specific API requirements above the requirements
    enforced in other languages (like Gen in Julia). In particular,
    any implementation must provide a :code:`__call__` method so that
    JAX can correctly determine output shapes.

    The user *must* match the interface signatures of the native JAX
    implementation. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.

    To support argument and choice gradients via JAX, the user must
    provide a differentiable `importance` implementation.
    """

    @abc.abstractmethod
    def __call__(self, key, *args):
        pass

    def get_trace_type(self, key, args, **kwargs):
        shape = kwargs.get("shape", ())
        return Bottom(shape)

    def simulate(self, key, args):
        pass

    def importance(self, key, chm, args):
        pass

    def update(self, key, original, new, args):
        pass

    def arg_grad(self, key, tr, args, argnums):
        pass

    def choice_grad(self, key, tr, chm, args):
        pass

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)


#####
# Trace
#####


@dataclass
class Trace(Pytree):
    @abc.abstractmethod
    def get_retval(self):
        pass

    @abc.abstractmethod
    def get_score(self):
        pass

    @abc.abstractmethod
    def get_args(self):
        pass

    @abc.abstractmethod
    def get_choices(self):
        pass

    @abc.abstractmethod
    def get_gen_fn(self):
        pass

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(
                    indent,
                    pp.concat(
                        [
                            gpp._pformat(self.get_gen_fn(), **kwargs),
                            pp.brk(),
                            gpp._pformat(self.get_choices(), **kwargs),
                        ]
                    ),
                ),
                pp.brk(),
                pp.text("return: "),
                gpp._pformat(self.get_retval(), **kwargs),
            ]
        )

    def has_choice(self, addr):
        choices = self.get_choices()
        return choices.has_choice(addr)

    def get_choice(self, addr):
        choices = self.get_choices()
        return choices.get_choice(addr)

    def has_value(self):
        choices = self.get_choices()
        return choices.has_value()

    def get_value(self):
        choices = self.get_choices()
        return choices.get_value()

    def get_choices_shallow(self):
        choices = self.get_choices()
        return choices.get_choices_shallow()

    def merge(self, other):
        return self.get_choices().merge(other.get_choices())

    def get_selection(self):
        return self.get_choices().get_selection()

    def strip_metadata(self):
        return self.get_choices().strip_metadata()

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choices = self.get_choices()
        choice = choices.get_choice(addr)
        if choice.has_value():
            return choice.get_value()
        else:
            return choice


#####
# ChoiceMap
#####


@dataclass
class ChoiceMap(Pytree):
    @abc.abstractmethod
    def has_choice(self, addr):
        pass

    @abc.abstractmethod
    def get_choice(self, addr):
        pass

    @abc.abstractmethod
    def has_value(self):
        pass

    @abc.abstractmethod
    def get_value(self):
        pass

    @abc.abstractmethod
    def get_choices_shallow(self):
        pass

    @abc.abstractmethod
    def merge(self, other):
        pass

    def overload_pprint(self, **kwargs):
        entries = []
        indent = kwargs["indent"]
        for (k, v) in self.get_choices_shallow():
            entry = gpp._dict_entry(k, v, **kwargs)
            entries.append(entry)
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(indent, pp.join(gpp._comma_sep, entries)),
            ]
        )

    def get_selection(self):
        return NoneSelection()

    def get_choices(self):
        return self

    def slice(self, arr: Sequence):
        def _inner(v):
            return v[arr]

        return squeeze(
            jtu.tree_map(
                _inner,
                self,
            )
        )

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choice = self.get_choice(addr)
        if choice.has_value():
            return choice.get_value()
        else:
            return choice

    def __eq__(self, other):
        return self.flatten() == other.flatten()


@dataclass
class EmptyChoiceMap(ChoiceMap):
    def flatten(self):
        return (), ()

    def has_choice(self, addr):
        return False

    def get_choice(self, addr):
        raise Exception("EmptyChoiceMap does not address any values.")

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("EmptyChoiceMap is not a value choice map.")

    def get_choices_shallow(self):
        return ()

    def merge(self, other):
        return other

    def strip_metadata(self):
        return self


@dataclass
class ValueChoiceMap(ChoiceMap):
    value: Any

    def flatten(self):
        return (self.value,), ()

    def overload_pprint(self, **kwargs):
        return pp.concat(
            [
                pp.text("(value = "),
                gpp._pformat(self.value, **kwargs),
                pp.text(")"),
            ]
        )

    def has_value(self):
        return True

    def get_value(self):
        return self.value

    def has_choice(self, *addr):
        return len(addr) == 0

    def get_choice(self, *addr):
        if len(addr) == 0:
            return self.value
        else:
            return EmptyChoiceMap()

    def get_choices_shallow(self):
        return [((), self.value)]

    def strip_metadata(self):
        return self

    def get_selection(self):
        return AllSelection()

    def merge(self, other):
        return other

    def __hash__(self):
        if isinstance(self.value, np.ndarray):
            return hash(self.value.tostring())
        else:
            return hash(self.value)


#####
# Selection
#####


@dataclass
class Selection(Pytree):
    @abc.abstractmethod
    def filter(self, chm):
        pass

    @abc.abstractmethod
    def complement(self):
        pass

    def get_selection(self):
        return self

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)


@dataclass
class NoneSelection(Selection):
    def flatten(self):
        return (), ()

    def filter(self, chm):
        return EmptyChoiceMap(), 0.0

    def complement(self):
        return AllSelection()


@dataclass
class AllSelection(Selection):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]):
        if isinstance(v, Trace):
            return v, v.get_score()
        else:
            return v, 0.0

    def complement(self):
        return NoneSelection()

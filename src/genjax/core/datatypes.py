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
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

import jax
import jax.tree_util as jtu
import numpy as np
import rich

import genjax.core.pretty_printing as gpp
from genjax.core.pytree import Pytree
from genjax.core.tracetypes import Bottom
from genjax.core.tracetypes import TraceType
from genjax.core.tree import Leaf
from genjax.core.tree import Tree
from genjax.core.typing import Bool
from genjax.core.typing import Float
from genjax.core.typing import PRNGKey


#####
# ChoiceMap
#####


@dataclass
class ChoiceMap(Tree):
    @abc.abstractmethod
    def get_selection(self):
        pass

    def get_choices(self):
        return self

    def strip(self):
        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self, is_leaf=_check)

    def __eq__(self, other):
        return self.flatten() == other.flatten()

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choice = self.get_subtree(addr)
        if isinstance(choice, Leaf):
            return choice.get_leaf_value()
        else:
            return choice


#####
# Trace
#####


@dataclass
class Trace(ChoiceMap, Tree):
    @abc.abstractmethod
    def get_retval(self) -> Any:
        pass

    @abc.abstractmethod
    def get_score(self) -> Float:
        pass

    @abc.abstractmethod
    def get_args(self) -> Tuple:
        pass

    @abc.abstractmethod
    def get_choices(self) -> ChoiceMap:
        pass

    @abc.abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        pass

    def has_subtree(self, addr) -> Bool:
        choices = self.get_choices()
        return choices.has_subtree(addr)

    def get_subtree(self, addr) -> ChoiceMap:
        choices = self.get_choices()
        return choices.get_subtree(addr)

    def get_subtrees_shallow(self):
        choices = self.get_choices()
        return choices.get_subtrees_shallow()

    def merge(self, other) -> ChoiceMap:
        return self.get_choices().merge(other.get_choices())

    def get_selection(self):
        return self.get_choices().get_selection()

    def strip(self):
        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self.get_choices(), is_leaf=_check)

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choices = self.get_choices()
        choice = choices.get_subtree(addr)
        if isinstance(choice, Leaf):
            return choice.get_leaf_value()
        else:
            return choice


#####
# Selection
#####


@dataclass
class Selection(Tree):
    @abc.abstractmethod
    def filter(self, chm):
        pass

    @abc.abstractmethod
    def complement(self):
        pass

    def get_selection(self):
        return self

    def __getitem__(self, addr):
        subselection = self.get_subtree(addr)
        return subselection


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
    def __call__(self, key: PRNGKey, *args):
        pass

    def get_trace_type(
        self,
        key: PRNGKey,
        args: Tuple,
        **kwargs,
    ) -> TraceType:
        shape = kwargs.get("shape", ())
        return Bottom(shape)

    @abc.abstractmethod
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[PRNGKey, Trace]:
        pass

    @abc.abstractmethod
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Float, Trace]]:
        pass

    @abc.abstractmethod
    def update(
        self,
        key: PRNGKey,
        original: Trace,
        new: ChoiceMap,
        diffs: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Any, Float, Trace, ChoiceMap]]:
        pass

    @abc.abstractmethod
    def assess(
        self,
        key: PRNGKey,
        evaluation_point: ChoiceMap,
        args: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Any, Float]]:
        pass

    def unzip(
        self,
        key: PRNGKey,
        fixed: ChoiceMap,
    ) -> Tuple[
        PRNGKey,
        Callable[[ChoiceMap, Tuple], Float],
        Callable[[ChoiceMap, Tuple], Any],
    ]:
        key, sub_key = jax.random.split(key)

        def score(provided, args) -> Float:
            merged = fixed.merge(provided)
            _, (_, score) = self.assess(sub_key, merged, args)
            return score

        def retval(provided, args) -> Any:
            merged = fixed.merge(provided)
            _, (retval, _) = self.assess(sub_key, merged, args)
            return retval

        return key, score, retval


#####
# Concrete choice maps
#####


@dataclass
class EmptyChoiceMap(ChoiceMap, Leaf):
    def flatten(self):
        return (), ()

    def get_leaf_value(self):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def get_selection(self):
        return NoneSelection()


@dataclass
class ValueChoiceMap(ChoiceMap, Leaf):
    value: Any

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def new(cls, v):
        if isinstance(v, ValueChoiceMap):
            return ValueChoiceMap.new(v.get_leaf_value())
        else:
            return ValueChoiceMap(v)

    def get_leaf_value(self):
        return self.value

    def get_selection(self):
        return AllSelection()

    def __hash__(self):
        if isinstance(self.value, np.ndarray):
            return hash(self.value.tobytes())
        else:
            return hash(self.value)

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        if isinstance(self.value, Pytree):
            subt = self.value._build_rich_tree()
            tree.add(subt)
        else:
            tree.add(gpp.tree_pformat(self.value))
        return tree


#####
# Concrete selections
#####


@dataclass
class NoneSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, chm):
        return EmptyChoiceMap(), 0.0

    def complement(self):
        return AllSelection()

    def get_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )


@dataclass
class AllSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]):
        if isinstance(v, Trace):
            return v, v.get_score()
        else:
            return v, 0.0

    def complement(self):
        return NoneSelection()

    def get_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )

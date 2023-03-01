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
import dataclasses
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

import jax
import jax.tree_util as jtu
import numpy as np

from genjax._src.core.datatypes.masks import BooleanMask
from genjax._src.core.datatypes.tracetypes import Bottom
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.tree import Leaf
from genjax._src.core.datatypes.tree import Tree
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.pytree import Pytree
from genjax._src.core.pytree import tree_grad_split
from genjax._src.core.pytree import tree_zipper
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey


#####
# ChoiceMap
#####


@dataclasses.dataclass
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
        choice = self.get_subtree(addr)
        if isinstance(choice, Leaf):
            v = choice.get_leaf_value()

            # If the choice is a Leaf, it might participate in masking.
            # Here, we check if the value is masked.
            # Then, we either unwrap the mask - or return it,
            # depending on the concreteness of the mask value.
            if isinstance(v, BooleanMask):
                if is_concrete(v.mask):
                    if v.mask:
                        return v.unmask()
                    else:
                        return EmptyChoiceMap()
                else:
                    return v
            else:
                return v
        else:
            return choice


#####
# Trace
#####


@dataclasses.dataclass
class Trace(ChoiceMap, Tree):
    @abc.abstractmethod
    def get_retval(self) -> Any:
        pass

    @abc.abstractmethod
    def get_score(self) -> FloatArray:
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

    @abc.abstractmethod
    def project(self, selection: "Selection") -> FloatArray:
        pass

    def update(self, key, choices, argdiffs):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    def has_subtree(self, addr) -> BoolArray:
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
        choices = self.get_choices()
        choice = choices.get_subtree(addr)
        if isinstance(choice, BooleanMask):
            if is_concrete(choice.mask):
                if choice.mask:
                    return choice.unmask()
                else:
                    return EmptyChoiceMap()
            else:
                return choice
        elif isinstance(choice, Leaf):
            return choice.get_leaf_value()
        else:
            return choice


#####
# Selection
#####


@dataclasses.dataclass
class Selection(Tree):
    @abc.abstractmethod
    def filter(self, chm: ChoiceMap) -> ChoiceMap:
        pass

    @abc.abstractmethod
    def complement(self) -> "Selection":
        pass

    def get_selection(self):
        return self

    def __getitem__(self, addr):
        subselection = self.get_subtree(addr)
        return subselection


#####
# Generative function
#####


@dataclasses.dataclass
class GenerativeFunction(Pytree):
    """Abstract class which provides an inheritance base for user-defined
    implementations of the generative function interface methods e.g. the
    `BuiltinGenerativeFunction` and `Distribution` languages both implement a
    class inheritor of `GenerativeFunction`.

    Any implementation will interact with the JAX tracing machinery,
    however, so there are specific API requirements above the requirements
    enforced in other languages (unlike Gen in Julia, for example).

    The user *must* match the interface signatures of the native JAX
    implementation. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.

    To support argument and choice gradients via JAX, the user must
    provide a differentiable `importance` implementation.
    """

    # This is used to support tracing -- the user is not required to provide
    # a PRNGKey, because the value of the key is not important, only
    # the fact that the value has type PRNGKey.
    def __abstract_call__(self, *args) -> Tuple[PRNGKey, Any]:
        key = jax.random.PRNGKey(0)
        _, tr = self.simulate(key, args)
        retval = tr.get_retval()
        return retval

    def get_trace_type(self, *args, **kwargs) -> TraceType:
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
    ) -> Tuple[PRNGKey, Tuple[FloatArray, Trace]]:
        pass

    @abc.abstractmethod
    def update(
        self,
        key: PRNGKey,
        original: Trace,
        new: ChoiceMap,
        diffs: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:
        pass

    @abc.abstractmethod
    def assess(
        self,
        key: PRNGKey,
        evaluation_point: ChoiceMap,
        args: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        pass

    def unzip(
        self,
        key: PRNGKey,
        fixed: ChoiceMap,
    ) -> Tuple[
        PRNGKey,
        Callable[[ChoiceMap, Tuple], FloatArray],
        Callable[[ChoiceMap, Tuple], Any],
    ]:
        key, sub_key = jax.random.split(key)

        def score(differentiable: Tuple, nondifferentiable: Tuple) -> FloatArray:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.merge(provided)
            _, (_, score) = self.assess(sub_key, merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.merge(provided)
            _, (retval, _) = self.assess(sub_key, merged, args)
            return retval

        return key, score, retval

    # A higher-level gradient API - it relies upon `unzip`,
    # but provides convenient access to first-order gradients.
    def choice_grad(self, key, trace, selection):
        fixed = selection.complement().filter(trace.strip())
        evaluation_point = selection.filter(trace.strip())
        key, scorer, _ = self.unzip(key, fixed)
        grad, nograd = tree_grad_split(
            (evaluation_point, trace.get_args()),
        )
        choice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)
        return key, choice_gradient_tree

    ###################
    # ADEV and fusion #
    ###################

    def adev_convert(self, key: PRNGKey, args: Tuple) -> Tuple[PRNGKey, Any]:
        """Convert a generative function to a forward sampling function which
        uses primitives which are compatible with ADEV's probabilistic
        computation language."""
        raise NotImplementedError

    def prepare_fuse(self, key: PRNGKey, args: Tuple):
        """Convert a generative function to a canonical form with ADEV
        primitives for proposal fusion."""
        raise NotImplementedError

    def fuse(self, _: "GenerativeFunction"):
        """Fuse a generative function and a proposal to produce a probabilistic
        computation that returns an ELBO estimate."""
        raise NotImplementedError


#####
# Concrete choice maps
#####


@dataclasses.dataclass
class EmptyChoiceMap(ChoiceMap, Leaf):
    def flatten(self):
        return (), ()

    # Overload Leaf: matches Gen.jl - getting a subtree from
    # empty returns empty.
    def get_subtree(self, addr):
        return self

    def get_leaf_value(self):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def set_leaf_value(self, v):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def get_selection(self):
        return NoneSelection()


@dataclasses.dataclass
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

    def set_leaf_value(self, v):
        return ValueChoiceMap(v)

    def get_selection(self):
        return AllSelection()

    def __hash__(self):
        if isinstance(self.value, np.ndarray):
            return hash(self.value.tobytes())
        else:
            return hash(self.value)


#####
# Concrete selections
#####


@dataclasses.dataclass
class NoneSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]) -> ChoiceMap:
        return EmptyChoiceMap()

    def complement(self):
        return AllSelection()

    def get_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )

    def set_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )


@dataclasses.dataclass
class AllSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]):
        return v.get_choices()

    def complement(self):
        return NoneSelection()

    def get_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )

    def set_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )


##############
# Shorthands #
##############

value_choice_map = ValueChoiceMap.new
val_chm = value_choice_map
all_select = AllSelection.new
all_sel = all_select
none_select = NoneSelection.new
none_sel = none_select

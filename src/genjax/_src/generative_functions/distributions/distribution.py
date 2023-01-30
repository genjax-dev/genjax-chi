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

"""This module contains the `Distribution` abstract base class."""

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core import mask
from genjax._src.core.datatypes import AllSelection
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.core.diff_rules import Diff
from genjax._src.core.diff_rules import NoChange
from genjax._src.core.diff_rules import check_is_diff
from genjax._src.core.diff_rules import check_no_change
from genjax._src.core.diff_rules import strip_diff
from genjax._src.core.masks import BooleanMask
from genjax._src.core.specialization import concrete_cond
from genjax._src.core.tracetypes import TraceType
from genjax._src.core.tree import Leaf
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import (
    DeferredGenerativeFunctionCall,
)
from genjax._src.generative_functions.builtin.builtin_tracetype import lift


#####
# DistributionTrace
#####


@dataclass
class DistributionTrace(Trace, Leaf):
    gen_fn: GenerativeFunction
    args: Tuple
    value: Any
    score: FloatArray

    def flatten(self):
        return (self.gen_fn, self.args, self.value, self.score), ()

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.value

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_choices(self):
        return ValueChoiceMap(self.value)

    def project(self, selection: Selection) -> FloatArray:
        if isinstance(selection, AllSelection):
            return self.get_score()
        else:
            return 0.0

    def get_leaf_value(self):
        return self.value

    def set_leaf_value(self, v):
        return DistributionTrace(self.gen_fn, self.args, v, self.score)


#####
# Distribution
#####


@dataclass
class Distribution(GenerativeFunction):
    def flatten(self):
        return (), ()

    # This overloads the call functionality for this generative function
    # and allows usage of shorthand notation in the builtin DSL.
    def __call__(self, *args, **kwargs) -> DeferredGenerativeFunctionCall:
        return DeferredGenerativeFunctionCall.new(self, args, kwargs)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        # `get_trace_type` is compile time - the key value
        # doesn't matter, just the type.
        key = jax.random.PRNGKey(1)
        _, (_, (_, ttype)) = jax.make_jaxpr(self.random_weighted, return_shape=True)(
            key, *args
        )
        return lift(ttype)

    @abc.abstractmethod
    def random_weighted(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def estimate_logpdf(cls, key, v, *args, **kwargs):
        pass

    @typecheck
    def simulate(
        self, key: PRNGKey, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, DistributionTrace]:
        key, (w, v) = self.random_weighted(key, *args, **kwargs)
        tr = DistributionTrace(self, args, v, w)
        return key, tr

    @typecheck
    def importance(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[FloatArray, DistributionTrace]]:
        assert isinstance(chm, Leaf)

        # If the choice map is empty, we just simulate
        # and return 0.0 for the log weight.
        if isinstance(chm, EmptyChoiceMap):
            key, tr = self.simulate(key, args, **kwargs)
            return key, (0.0, tr)

        # If it's not empty, we should check if it is a mask.
        # If it is a mask, we need to see if it is active or not,
        # and then unwrap it - and use the active flag to determine
        # what to do at runtime.
        v = chm.get_leaf_value()
        if isinstance(v, BooleanMask):
            active = v.mask
            v = v.unmask()

            def _active(key, v, args):
                key, (w, v) = self.estimate_logpdf(key, v, *args)
                return key, v, w

            def _inactive(key, v, _):
                w = 0.0
                return key, v, w

            key, v, w = concrete_cond(
                active,
                _active,
                _inactive,
                key,
                v,
                args,
            )
            score = w

        # Otherwise, we just estimate the logpdf of the value
        # we got out of the choice map.
        else:
            key, (w, v) = self.estimate_logpdf(key, v, *args)
            score = w

        return key, (
            w,
            DistributionTrace(self, args, v, score),
        )

    # NOTE: Here's an interesting note about `update`...
    # (really, any of the GFI methods for any generative function)
    # - they should return homogeneous types for any return
    # branch leading out of the call.
    # Because these methods may be invoked in `jax.lax.switch` calls
    # it's important that callers have some knowledge about the
    # consistency of invoking a callee -- most generative function
    # languages ensure this is true by default e.g. if they defer
    # some of their behavior to callees.
    # For `Distribution` this is not true by default - we have to be
    # careful when defining the methods, and this is most true of update
    # below.
    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: DistributionTrace,
        constraints: ChoiceMap,
        argdiffs: Tuple,
        **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, DistributionTrace, Any]]:
        assert isinstance(constraints, Leaf)
        maybe_discard = mask(False, prev.get_choices())

        # Incremental optimization - if nothing has changed,
        # just return the previous trace.
        if isinstance(constraints, EmptyChoiceMap) and all(
            map(check_no_change, argdiffs)
        ):
            v = prev.get_retval()
            retval_diff = Diff.new(v, change=NoChange)
            return key, (retval_diff, 0.0, prev, maybe_discard)

        # Otherwise, we consider the cases.
        args = jtu.tree_map(strip_diff, argdiffs, is_leaf=check_is_diff)

        # First, we have to check if the trace provided
        # is masked or not. It's possible that a trace
        # with a mask is updated.
        prev_v = prev.get_retval()
        active = True
        if isinstance(prev_v, BooleanMask):
            active = prev_v.mask
            prev_v = prev_v.unmask()

        # Case 1: the new choice map is empty here.
        if isinstance(constraints, EmptyChoiceMap):
            prev_score = prev.get_score()
            v = prev_v

            # If the value is active, we compute any weight
            # corrections from changing arguments.
            def _active(key, v, *args):
                key, (fwd, _) = self.estimate_logpdf(key, v, *args)
                return key, fwd - prev_score

            # If the value is inactive, we do nothing.
            def _inactive(key, v, *args):
                return key, prev_score

            key, w = concrete_cond(active, _active, _inactive, key, v, *args)
            discard = maybe_discard
            retval_diff = jtu.tree_map(lambda v: Diff.new(v, change=NoChange), prev_v)

        # Case 2: the new choice map is not empty here.
        else:
            prev_score = prev.get_score()
            v = constraints.get_leaf_value()

            # Now, we must check if the choice map has a masked
            # leaf value, and dispatch accordingly.
            active_chm = True
            if isinstance(v, BooleanMask):
                active_chm = v.mask
                v = v.unmask()

            # The only time this flag is on is when both leaf values
            # are concrete, or they are both masked with true mask
            # values.
            active = jnp.all(jnp.logical_and(active_chm, active))

            def _constraints_active(key, v, *args):
                key, (fwd, _) = self.estimate_logpdf(key, v, *args)
                return key, v, fwd - prev_score

            def _constraints_inactive(key, v, *args):
                return key, prev_v, 0.0

            key, v, w = concrete_cond(
                active_chm, _constraints_active, _constraints_inactive, key, v, *args
            )

            discard = mask(active_chm, ValueChoiceMap(prev.get_leaf_value()))
            retval_diff = Diff.new(v)

        return key, (
            retval_diff,
            w,
            DistributionTrace(self, args, v, w),
            discard,
        )

    @typecheck
    def assess(
        self, key: PRNGKey, evaluation_point: ValueChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        v = evaluation_point.get_leaf_value()
        key, (score, _) = self.estimate_logpdf(key, v, *args)
        return key, (v, score)


#####
# ExactDensity
#####


@dataclass
class ExactDensity(Distribution):
    @abc.abstractmethod
    def sample(self, key, *args, **kwargs):
        pass

    @abc.abstractmethod
    def logpdf(self, v, *args, **kwargs):
        pass

    def random_weighted(self, key, *args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *args, **kwargs)
        w = self.logpdf(v, *args, **kwargs)
        return key, (w, v)

    def estimate_logpdf(self, key, v, *args, **kwargs):
        w = self.logpdf(v, *args, **kwargs)
        return key, (w, v)

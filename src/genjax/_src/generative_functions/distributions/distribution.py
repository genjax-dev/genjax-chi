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
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import Leaf
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.masking import Mask
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.datatypes.tracetypes import tt_lift
from genjax._src.core.interpreters.staging import concrete_cond
from genjax._src.core.transforms.incremental import Diff
from genjax._src.core.transforms.incremental import NoChange
from genjax._src.core.transforms.incremental import UnknownChange
from genjax._src.core.transforms.incremental import diff
from genjax._src.core.transforms.incremental import static_check_no_change
from genjax._src.core.transforms.incremental import static_check_tree_leaves_diff
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import SupportsBuiltinSugar


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
class Distribution(JAXGenerativeFunction, SupportsBuiltinSugar):
    def flatten(self):
        return (), ()

    def __abstract_call__(self, *args):
        # Abstract evaluation: value here doesn't matter, only the type.
        key = jax.random.PRNGKey(0)
        _, (_, v) = self.random_weighted(key, *args)
        return v

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        # `get_trace_type` is compile time - the key value
        # doesn't matter, just the type.
        key = jax.random.PRNGKey(1)
        _, (_, (_, ttype)) = jax.make_jaxpr(self.random_weighted, return_shape=True)(
            key, *args
        )
        return tt_lift(ttype)

    @abc.abstractmethod
    def random_weighted(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def estimate_logpdf(self, key, v, *args, **kwargs):
        pass

    @typecheck
    def simulate(
        self, key: PRNGKey, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, DistributionTrace]:
        key, (w, v) = self.random_weighted(key, *args, **kwargs)
        tr = DistributionTrace(self, args, v, w)
        return key, tr

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: EmptyChoiceMap,
        args: Tuple,
        **kwargs,
    ) -> Tuple[PRNGKey, Tuple[FloatArray, DistributionTrace]]:
        key, tr = self.simulate(key, args, **kwargs)
        return key, (0.0, tr)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ValueChoiceMap,
        args: Tuple,
        **kwargs,
    ) -> Tuple[PRNGKey, Tuple[FloatArray, DistributionTrace]]:

        # If it's not empty, we should check if it is a mask.
        # If it is a mask, we need to see if it is active or not,
        # and then unwrap it - and use the active flag to determine
        # what to do at runtime.
        v = chm.get_leaf_value()
        if isinstance(v, Mask):
            active = v.mask
            v = v.unmask()

            def _active(key, v, args):
                key, w = self.estimate_logpdf(key, v, *args)
                return key, v, w

            def _inactive(key, v, _):
                w = 0.0
                key, (_, v) = self.random_weighted(key, *args)
                return key, v, w

            key, v, w = concrete_cond(active, _active, _inactive, key, v, args)
            score = w

        # Otherwise, we just estimate the logpdf of the value
        # we got out of the choice map.
        else:
            key, w = self.estimate_logpdf(key, v, *args)
            score = w

        return key, (
            w,
            DistributionTrace(self, args, v, score),
        )

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: DistributionTrace,
        constraints: EmptyChoiceMap,
        argdiffs: Tuple,
        **kwargs,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, DistributionTrace, Any]]:
        static_check_tree_leaves_diff(argdiffs)
        v = prev.get_retval()
        retval_diff = jtu.tree_map(lambda v: Diff(v, NoChange), v)
        discard = mask(False, prev.get_choices())

        # If no change to arguments, no need to update.
        if static_check_no_change(argdiffs):
            return key, (retval_diff, 0.0, prev, discard)

        # Otherwise, we must compute an incremental weight.
        else:
            args = tree_diff_primal(argdiffs)
            key, fwd = self.estimate_logpdf(key, v, *args)
            bwd = prev.get_score()
            new_tr = DistributionTrace(self, args, v, fwd)
            return key, (retval_diff, fwd - bwd, new_tr, discard)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: DistributionTrace,
        constraints: ValueChoiceMap,
        argdiffs: Tuple,
        **kwargs,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, DistributionTrace, Any]]:
        static_check_tree_leaves_diff(argdiffs)
        args = tree_diff_primal(argdiffs)
        v = constraints.get_leaf_value()
        if isinstance(v, Mask):
            check, value = v.mask, v.value

            def _fallback(key):
                key, (retdiff, w, new_tr, discard) = self.update(
                    key, prev, EmptyChoiceMap(), argdiffs
                )
                primal = tree_diff_primal(retdiff)
                # Because we are pushing the update to depend dynamically on a mask flag value,
                # we have to ensure that all branches return the same types of Pytree.
                coerce_to_unknown = jtu.tree_map(
                    lambda v: diff(v, UnknownChange), primal
                )
                return key, (coerce_to_unknown, w, new_tr, discard)

            def _active(key):
                return self.update(key, prev, ValueChoiceMap(value), argdiffs)

            return concrete_cond(check, _active, _fallback, key)
        else:
            key, fwd = self.estimate_logpdf(key, v, *args)
            bwd = prev.get_score()
            w = fwd - bwd
            new_tr = DistributionTrace(self, args, v, fwd)
            discard = mask(True, prev.get_choices())
            retval_diff = jtu.tree_map(lambda x: diff(x, UnknownChange), v)

            return key, (retval_diff, w, new_tr, discard)

    @typecheck
    def assess(
        self, key: PRNGKey, evaluation_point: ValueChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        v = evaluation_point.get_leaf_value()
        key, score = self.estimate_logpdf(key, v, *args)
        return key, (v, score)


#####
# ExactDensity
#####


@dataclass
class ExactDensity(Distribution):
    """> Abstract base class which extends Distribution and assumes that the
    implementor provides an exact logpdf method (compared to one which returns
    _an estimate of the logpdf_).

    All of the standard distributions inherit from `ExactDensity`, and
    if you are looking to implement your own distribution, you should
    likely use this class.

    !!! info "`Distribution` implementors are `Pytree` implementors"

    As `Distribution` extends `Pytree`, if you use this class, you must
    implement `flatten` as part of your class declaration.
    """

    @abc.abstractmethod
    def sample(self, key: PRNGKey, *args, **kwargs) -> Any:
        """> Sample from the distribution, returning a value from the event
        space.

        Arguments:
            key: A `PRNGKey`.
            *args: The arguments to the distribution invocation.

        Returns:
            v: A value from the event space of the distribution.

        !!! info "Implementations need not return a new `PRNGKey`"

            Note that `sample` does not return a new evolved `PRNGKey`. This is for convenience - `ExactDensity` is used often, and the interface for `sample` is simple. `sample` is called by `random_weighted` in the generative function interface implementations, and always gets a fresh `PRNGKey` - `sample` as a callee need not return a new evolved key.

        Examples:
            `genjax.normal` is a distribution with an exact density, which supports the `sample` interface. Here's an example of invoking `sample`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            v = genjax.normal.sample(key, 0.0, 1.0)
            print(console.render(v))
            ```

            Note that you often do want or need to invoke `sample` directly - you'll likely want to use the generative function interface methods instead:

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            key, tr = genjax.normal.simulate(key, (0.0, 1.0))
            print(console.render(tr))
            ```
        """

    @abc.abstractmethod
    def logpdf(self, v, *args, **kwargs):
        """> Given a value from the event space, compute the log probability of
        that value under the distribution."""

    def random_weighted(self, key, *args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *args, **kwargs)
        w = self.logpdf(v, *args, **kwargs)
        return key, (w, v)

    def estimate_logpdf(self, key, v, *args, **kwargs):
        w = self.logpdf(v, *args, **kwargs)
        return key, w


#####
# Product
#####


@dataclass
class Product(Distribution):
    components: List[Distribution]

    def flatten(self):
        return (self.components,), ()

    @classmethod
    def new(cls, *other: Distribution):
        v = Product([])
        for sub in other:
            v.append(sub)
        return v

    def append(self, other: Distribution):
        n = Product(self.components)
        if isinstance(other, Product):
            for sub in other.components:
                n.append(sub)
        else:
            n.components.append(other)
        return n

    @typecheck
    def random_weighted(self, key: PRNGKey, *args):
        assert len(args) == len(self.components)
        tw, ret = 0.0, []
        for (op_args, op) in zip(args, self.components):
            key, (w, v) = op.random_weighted(key, *op_args)
            tw += w
            ret.append(v)
        return key, (tw, (*ret,))

    @typecheck
    def estimate_logpdf(self, key: PRNGKey, v: Tuple, *args):
        assert len(args) == len(self.components)
        tw = 0.0
        for (op, op_args, r) in zip(self.components, args, v):
            key, w = op.estimate_logpdf(key, r, *op_args)
            tw += w
        return key, tw

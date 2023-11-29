# Copyright 2023 MIT Probabilistic Computing Project
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

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceValue
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.interpreters.incremental import static_check_no_change
from genjax._src.core.interpreters.incremental import static_check_tree_leaves_diff
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_unknown_change
from genjax._src.core.serialization.pickle import PickleDataFormat
from genjax._src.core.serialization.pickle import PickleSerializationBackend
from genjax._src.core.serialization.pickle import SupportsPickleSerialization
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


#####
# DistributionTrace
#####


@dataclass
class DistributionTrace(
    Trace,
    SupportsPickleSerialization,
):
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
        return ChoiceValue(self.value)

    def project(self, selection: Selection) -> FloatArray:
        if isinstance(selection, AllSelection):
            return self.get_score()
        else:
            return 0.0

    def get_value(self):
        return self.value

    def set_leaf_value(self, v):
        return DistributionTrace(self.gen_fn, self.args, v, self.score)

    #################
    # Serialization #
    #################

    @dispatch
    def dumps(
        self,
        backend: PickleSerializationBackend,
    ) -> PickleDataFormat:
        args, value, score = self.args, self.value, self.score
        payload = [
            backend.dumps(args),
            backend.dumps(value),
            backend.dumps(score),
        ]
        return PickleDataFormat(payload)


#####
# Distribution
#####


@dataclass
class Distribution(GenerativeFunction, SupportsCalleeSugar):
    def flatten(self):
        return (), ()

    @abc.abstractmethod
    def random_weighted(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def estimate_logpdf(self, key, v, *args, **kwargs):
        pass

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DistributionTrace:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: EmptyChoice,
        args: Tuple,
    ) -> Tuple[DistributionTrace, FloatArray]:
        tr = self.simulate(key, args)
        return (tr, 0.0)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceValue,
        args: Tuple,
    ) -> Tuple[DistributionTrace, FloatArray]:
        v = chm.get_value()
        w = self.estimate_logpdf(key, v, *args)
        score = w
        return (DistributionTrace(self, args, v, score), w)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: DistributionTrace,
        constraints: EmptyChoice,
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, FloatArray, Any, Any]:
        static_check_tree_leaves_diff(argdiffs)
        v = prev.get_retval()
        retval_diff = tree_diff_no_change(v)

        # If no change to arguments, no need to update.
        if static_check_no_change(argdiffs):
            return (prev, 0.0, retval_diff, EmptyChoice())

        # Otherwise, we must compute an incremental weight.
        else:
            args = tree_diff_primal(argdiffs)
            fwd = self.estimate_logpdf(key, v, *args)
            bwd = prev.get_score()
            new_tr = DistributionTrace(self, args, v, fwd)
            return (new_tr, fwd - bwd, retval_diff, EmptyChoice())

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: DistributionTrace,
        constraints: ChoiceValue,
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, FloatArray, Any, Any]:
        static_check_tree_leaves_diff(argdiffs)
        args = tree_diff_primal(argdiffs)
        v = constraints.get_value()
        fwd = self.estimate_logpdf(key, v, *args)
        bwd = prev.get_score()
        w = fwd - bwd
        new_tr = DistributionTrace(self, args, v, fwd)
        discard = prev.get_choices()
        retval_diff = tree_diff_unknown_change(v)
        return (new_tr, w, retval_diff, discard)

    ###################
    # Deserialization #
    ###################

    @dispatch
    def loads(
        self,
        data: PickleDataFormat,
        backend: PickleSerializationBackend,
    ) -> DistributionTrace:
        args, value, score = backend.loads(data.payload)
        return DistributionTrace(self, args, value, score)


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

        As `Distribution` extends `Pytree`, if you use this class, you must implement `flatten` as part of your class declaration.
    """

    @abc.abstractmethod
    def sample(self, key: PRNGKey, *args: Any, **kwargs) -> Any:
        """> Sample from the distribution, returning a value from the event
        space.

        Arguments:
            key: A `PRNGKey`.
            *args: The arguments to the distribution invocation.

        Returns:
            v: A value from the support of the distribution.

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
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            print(console.render(tr))
            ```
        """

    @abc.abstractmethod
    def logpdf(self, v: Any, *args: Any, **kwargs) -> FloatArray:
        """> Given a value from the support of the distribution, compute the
        log probability of that value under the density (with respect to the
        standard base measure).

        Arguments:
            v: A value from the support of the distribution.
            *args: The arguments to the distribution invocation.

        Returns:
            logpdf: The log density evaluated at `v`, with density configured by `args`.
        """

    def random_weighted(self, key, *args, **kwargs):
        v = self.sample(key, *args, **kwargs)
        w = self.logpdf(v, *args, **kwargs)
        return (w, v)

    def estimate_logpdf(self, _, v, *args, **kwargs):
        w = self.logpdf(v, *args, **kwargs)
        return w

    @typecheck
    def assess(
        self,
        evaluation_point: ChoiceValue,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        v = evaluation_point.get_value()
        score = self.logpdf(v, *args)
        return (score, v)

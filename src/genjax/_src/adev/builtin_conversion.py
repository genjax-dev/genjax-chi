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

import jax.tree_util as jtu

import genjax._src.core.interpreters.context as context
from genjax._src.adev.lang import lang as adev_lang
from genjax._src.adev.lang import sample as adev_sample
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.generative_functions.builtin.builtin_gen_fn import (
    BuiltinGenerativeFunction,
)
from genjax._src.generative_functions.builtin.builtin_transforms import (
    BuiltinInterfaceContext,
)


#####
# Probabilistic computation transform
#####


@dataclasses.dataclass
class ADEVConvertContext(BuiltinInterfaceContext):
    key: PRNGKey

    def flatten(self):
        return (self.key,), ()

    def yield_state(self):
        return (self.key,)

    @classmethod
    def new(cls, key):
        return ADEVConvertContext(
            key,
        )

    def handle_trace(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        args = tuple(args)
        adev_term = adev_lang(gen_fn)
        self.key, v = adev_sample(adev_term, self.key, args)
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def adev_conversion_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        ctx = ADEVConvertContext.new(key)
        retvals, statefuls = context.transform(source_fn, ctx)(*args, **kwargs)
        (key,) = statefuls
        return key, retvals

    return wrapper


@dispatch
def adev_convert(gen_fn: BuiltinGenerativeFunction):
    def adev_simulate(key: PRNGKey, args: Tuple, **kwargs):
        key, v = adev_conversion_transform(gen_fn.source, **kwargs)(key, args)
        return key, v

    return adev_simulate

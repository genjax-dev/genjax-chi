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

"""
This module provides an abstract interpreter which performs choice map
shape analysis on JAX generative function source code.
"""

import jax
from genjax.interface import simulate
from genjax.core.datatypes import GenerativeFunction


def trace_shape(f: GenerativeFunction):
    def __inner(f, *args):
        _, form = jax.make_jaxpr(simulate(f), return_shape=True)(*args)
        trace = form[1]
        values, tr_treedef = jax.tree_util.tree_flatten(trace)
        return values, tr_treedef, trace

    return lambda *args: __inner(f, *args)


def trace_shape_no_toplevel(f: GenerativeFunction):
    def __inner(f, *args):
        _, form = jax.make_jaxpr(simulate(f), return_shape=True)(*args)
        trace = form[1]
        chm = trace.get_choices()
        values, chm_treedef = jax.tree_util.tree_flatten(chm)
        return values, chm_treedef, chm

    return lambda *args: __inner(f, *args)


def choice_map_shape(f: GenerativeFunction):
    def __inner(f, *args):
        _, form = jax.make_jaxpr(simulate(f), return_shape=True)(*args)
        trace = form[1]
        chm = trace.get_choices()
        chm = chm.strip_metadata()
        values, chm_treedef = jax.tree_util.tree_flatten(chm)
        return values, chm_treedef, chm

    return lambda *args: __inner(f, *args)

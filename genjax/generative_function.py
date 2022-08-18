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

from genjax.handlers import (
    Sample,
    Simulate,
    Importance,
    Diff,
    Update,
    ArgumentGradients,
    ChoiceGradients,
)
from genjax.datatypes import Trace
from genjax.encapsulated import EncapsulatedGenerativeFunction
from jax import tree_util


#####
# Generative function interface
#####


def sample(f):
    def _inner(key, *args):
        fn = Sample().transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        k, *v = fn(key, *in_args)
        return k, (*v,)

    return lambda key, *args: _inner(key, *args)


def simulate(f):
    def _inner(key, args):
        fn = Simulate().transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        key, (r, chm, score) = fn(key, *in_args)
        return key, Trace(args, tuple(r), chm, score)

    # If `f` is an encapsulated generative function, pass
    # the call to the method.
    if isinstance(f, EncapsulatedGenerativeFunction):
        return lambda key, args: f.simulate(key, args)
    else:
        return lambda key, args: _inner(key, args)


def importance(f):
    def _inner(key, chm, args):
        fn = Importance(chm).transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        key, (w, r, chm, score) = fn(key, *in_args)
        return key, (w, Trace(args, tuple(r), chm, score))

    return lambda key, chm, args: _inner(key, chm, args)


def diff(f):
    def _inner(key, original, new, args):
        fn = Diff(original, new).transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        key, (w, ret) = fn(key, *in_args)
        return key, (w, ret)

    return lambda key, original, new, args: _inner(key, original, new, args)


def update(f):
    def _inner(key, original, new, args):
        fn = Update(original, new).transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        key, (w, ret, chm) = fn(key, *in_args)
        old = original.get_choices()
        discard = old.setdiff(chm)
        return key, (
            w,
            Trace(args, tuple(ret), chm, original.get_score() + w),
            discard,
        )

    return lambda key, chm, new, args: _inner(key, chm, new, args)


def arg_grad(f, argnums):
    def _inner(key, tr, argnums, args):
        fn = ArgumentGradients(tr, argnums).transform(f)(key, *args)
        in_args, _ = tree_util.tree_flatten(args)
        arg_grads, key = fn(key, *in_args)
        return key, arg_grads

    return lambda key, tr, args: _inner(key, tr, argnums, args)


def choice_grad(f):
    def _inner(key, tr, chm, args):
        fn = ChoiceGradients(tr).transform(f)(key, *args)
        choice_grads, key = fn(key, chm)
        return key, choice_grads

    return lambda key, tr, chm, args: _inner(key, tr, chm, args)

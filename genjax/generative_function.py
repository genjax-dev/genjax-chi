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


#####
# Generative function interface
#####


def sample(f):
    def _inner(*args):
        jitted = Sample().jit(f)(*args)
        k, *v = jitted(*args)
        return k, v

    return lambda *args: _inner(*args)


def simulate(f):
    def _inner(*args):
        jitted = Simulate().jit(f)(*args)
        (r, chm, score) = jitted(*args)
        return Trace(args, r, chm, score)

    return lambda *args: _inner(*args)


def importance(f):
    def _inner(chm, *args):
        jitted = Importance(chm).jit(f)(*args)
        (w, r, chm, score) = jitted(*args)
        return w, Trace(args, r, chm, score)

    return lambda chm, *args: _inner(chm, *args)


def diff(f):
    def _inner(original, new, *args):
        jitted = Diff(original, new).jit(f)(*args)
        w, ret = jitted(*args)
        return w, ret

    return lambda chm, *args: _inner(chm, *args)


def update(f):
    def _inner(original, new, *args):
        jitted = Update(original, new).jit(f)(*args)
        w, ret, chm = jitted(*args)
        return w, Trace(args, ret, chm, original.get_score() + w)

    return lambda chm, *args: _inner(chm, *args)


def arg_grad(f, argnums):
    def _inner(tr, argnums, *args):
        jitted = ArgumentGradients(tr, argnums).jit(f)(*args)
        arg_grads = jitted(*args)
        return arg_grads

    return lambda tr, *args: _inner(tr, argnums, *args)


def choice_grad(f):
    def _inner(tr, chm, *args):
        jitted = ChoiceGradients(tr).jit(f)(*args)
        choice_grads = jitted(chm)
        return choice_grads

    return lambda tr, chm, *args: _inner(tr, chm, *args)

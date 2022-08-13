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

from genjax.core.transforms import I
from genjax.handlers import (
    Sample,
    Simulate,
    Importance,
    Update,
    ArgumentGradients,
    ChoiceGradients,
)

from jax.tree_util import register_pytree_node


class Trace:
    def __init__(self, args, retval, choices, scores, score):
        self.args = args
        self.retval = retval
        self.choices = choices
        self.scores = scores
        self.score = score

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


register_pytree_node(
    Trace,
    lambda trace: (
        (trace.args, trace.retval, trace.choices, trace.scores, trace.score),
        None,
    ),
    lambda _, args: Trace(*args),
)


def sample(f):
    def _inner(*args):
        jitted = Sample().jit(f)(*args)
        k, *v = jitted(*args)
        return k, v

    return lambda *args: _inner(*args)


def simulate(f):
    def _inner(*args):
        jitted = Simulate().jit(f)(*args)
        (r, chm, scores, score) = jitted(*args)
        return Trace(args, r, chm, scores, score)

    return lambda *args: _inner(*args)


def importance(f):
    def _inner(chm, *args):
        jitted = Importance(chm).jit(f)(*args)
        (w, r, chm, scores, score) = jitted(*args)
        return w, Trace(args, r, chm, scores, score)

    return lambda chm, *args: _inner(chm, *args)


def update(f):
    def _inner(original, new, *args):
        jitted = Update(original, new).jit(f)(*args)
        w, ret, scores, chm = jitted(*args)
        return Trace(args, ret, chm, scores, original.get_score() + w), w

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

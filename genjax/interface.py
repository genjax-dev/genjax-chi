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
This module exposes the generative function interface -- a set of methods
defined for generative functions which support the implementation of
programmable inference algorithms.

Combined with the trace and choice map datatypes, these interface methods
are the conceptual core of generative functions.
"""


def sample(f, **kwargs):
    """
    `sample` runs the generative function forward, returning a new
    `PRNGKey` and a return value.
    """
    return lambda *args: f.simulate(*args, **kwargs)


def simulate(f, **kwargs):
    return lambda *args: f.simulate(*args, **kwargs)


def importance(f, **kwargs):
    return lambda *args: f.importance(*args, **kwargs)


def diff(f, **kwargs):
    return lambda *args: f.diff(*args, **kwargs)


def update(f, **kwargs):
    return lambda *args: f.update(*args, **kwargs)


def arg_grad(f, argnums, **kwargs):
    return lambda *args: f.arg_grad(argnums)(*args, **kwargs)


def choice_grad(f, **kwargs):
    return lambda *args: f.choice_grad(*args, **kwargs)

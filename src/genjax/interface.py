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

"""The generative function interface is a set of methods and associated types
defined for an implementor which support the generic construction (via
interface abstraction) of programmable inference algorithms and differentiable
programming.

Combined with the trace and choice map associated datatypes, the generative function interface
methods form the conceptual core of the computational behavior of generative functions.

.. note::

    This module exposes the generative function interface as a set of
    Python functions. When called with :code:`f: GenerativeFunction`
    and :code:`**kwargs`, they return the corresponding
    :code:`GenerativeFunction` method.

    Here's an example:

    .. jupyter-execute::

        import genjax
        fn = genjax.simulate(genjax.Normal)
        print(fn)

    If you know you have a :code:`GenerativeFunction`, you can just refer to the
    methods directly - but sometimes it is useful to use the getter variants
    (there's no runtime cost when using the getter variants in jitted code, JAX eliminates it).
"""

from typing import Callable


def simulate(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.simulate(*args, **kwargs)


def importance(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.importance(*args, **kwargs)


def update(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.update(*args, **kwargs)


def assess(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.assess(*args, **kwargs)


def unzip(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.unzip(*args, **kwargs)


def get_trace_type(gen_fn, **kwargs) -> Callable:
    return lambda *args: gen_fn.get_trace_type(*args, **kwargs)

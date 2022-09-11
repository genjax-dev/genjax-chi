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
The generative function interface is a set of methods defined for
generative functions which support the implementation of
programmable inference algorithms.

Combined with the trace and choice map datatypes, these interface methods
are the conceptual core of generative functions.

This module exposes the generative function interface as a set of generic
Python functions. When called with :code:`f: GenerativeFunction`
and :code:`**kwargs`, they return the corresponding
:code:`GenerativeFunction` method.
"""


def simulate(f, **kwargs):
    """
    :code:`simulate` accepts a function :code:`f` and
    returns a transformed function which implements the below semantics.

    Given :code:`key: PRNGKey` and :code:`args: Tuple`, sample
    :math:`t\sim p(\cdot;x)` and :math:`r\sim p(\cdot;args, t)` and
    apply the return value function :math:`ret = f(args, t)`.

    Compute the score of the sample :math:`t` under :math:`p(\cdot; x)`.

    Return the return value :math:`ret`, the sample :math:`t`,
    and the score in a :code:`Trace` instance, along with an evolved
    :code:`PRNGKey`.

    Parameters
    ----------
    key: :code:`jax.random.PRNGKey`
        A JAX-compatible PRNGKey.

    args: :code:`tuple`
        A tuple of argument values.

    Returns
    -------
    key: :code:`PRNGKey`
        An updated :code:`jax.random.PRNGKey`.

    tr: :code:`genjax.Trace`
        A representation of the recorded random choices, as well as
        inference metadata, accrued during the call.

    Example
    -------

    .. jupyter-execute::

        import jax
        import genjax

        @genjax.gen
        def model(key):
            key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
            return key, x

        key = jax.random.PRNGKey(314159)
        key, tr = genjax.simulate(model)(key, ())
        print(tr)
    """
    return lambda *args: f.simulate(*args, **kwargs)


def importance(f, **kwargs):
    return lambda *args: f.importance(*args, **kwargs)


def update(f, **kwargs):
    return lambda *args: f.update(*args, **kwargs)


def arg_grad(f, argnums, **kwargs):
    return lambda *args: f.arg_grad(argnums)(*args, **kwargs)


def choice_grad(f, **kwargs):
    return lambda *args: f.choice_grad(*args, **kwargs)


def get_trace_type(f, **kwargs):
    return lambda *args: f.get_trace_type(*args, **kwargs)

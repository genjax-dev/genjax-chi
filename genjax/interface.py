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
The generative function interface is a set of methods and associated types
defined for an implementor which support the generic construction (via interface abstraction)
of programmable inference algorithms and differentiable programming.

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


def simulate(f, **kwargs):
    """
    Given a :code:`key: jax.random.PRNGKey` and :code:`args: Tuple`, perform the
    following steps:

    * Sample :math:`t\sim p(\cdot;x)` and :math:`r\sim p(\cdot;x, t)`.
    * Compute the score of the sample :math:`t` under :math:`p(\cdot; x)`.
    * Return the return value :math:`r`, the choice samples :math:`t`, and the score :math:`s`
      in a :code:`Trace` instance (including the arguments :math:`x`) :math:`(x, r, t, s)`,
      along with an evolved :code:`jax.random.PRNGKey`.

    Parameters
    ----------
    key: :code:`jax.random.PRNGKey`
        A JAX-compatible pseudo-random number generator key.

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

        # Usage here.
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
    """
    Given :code:`key: jax.random.PRNGKey` and :code:`args: Tuple`, compute the
    trace type for a generative function - a static piece of data
    characterizing the internal choice map shape (and types) and return type.

    Parameters
    ----------
    key: :code:`jax.random.PRNGKey`
        A JAX-compatible pseudo-random number generator key.

    args: :code:`tuple`
        A tuple of argument values.

    Returns
    -------
    trace_type: :code:`genjax.TraceType`
        A static type representation of the internal structure of random choices in
        a generative function, including the return type.

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

        # Usage here.
        trace_type = genjax.get_trace_type(model)(key, ())

        print(trace_type)
    """
    return lambda *args: f.get_trace_type(*args, **kwargs)

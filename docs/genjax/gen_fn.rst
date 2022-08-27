What is a generative function? 
==============================

A generative function is a computational object which supports a concise
set of interfaces designed to support customizable Bayesian inference 
(*programmable inference*).

Formally, generative functions are mathematical representation of probabilistic
models that are expressive enough to permit models with random structure,
including capturing notions of variable existence uncertainty. This is described in `Marco Cusumano-Towner's thesis`_.

One useful reference implementation of these objects lies in `Gen.jl`_,
an encoding of generative functions and inference in Julia. In :code:`GenJAX`,
our implementation of these objects is akin to the :code:`static modeling language` of `Gen.jl`_ - we rely upon JAX to provide us with a useful intermediate representation for programs that we operate on using transformations.

Speaking of the interface above, if you'd like to jump right to reading about this, visit :doc:`interface`.

What do generative functions look like in GenJAX?
-------------------------------------------------

A generative function in :code:`GenJAX` looks like a pure Python function,
roughly the subset of Python acceptable by JAX.

.. jupyter-execute::

    import jax
    import genjax

    @genjax.gen
    def model(key):
        key, x = genjax.trace("x", genjax.Normal)(key, ())
        return key, x

    print(model)

This decorator returns a :code:`JAXGenerativeFunction` - a generative function
which implements the :doc:`interface` by utilizing JAX's tracing and program
transformation abilities.

Let's study the JAX representation of the code for this object.

.. jupyter-execute::
    
    key = jax.random.PRNGKey(314159)
    print(jax.make_jaxpr(model)(key))

In this lowered form, we can see that our JAX representation has a call to something called :code:`trace`, with a few constant keyword arguments.

JAX doesn't natively know how to handle :code:`trace` - it's a primitive that we've defined to support the generative function interface semantics.

We can utilize the interface to transform this representation to implement the semantics.

.. jupyter-execute::
    
    key = jax.random.PRNGKey(314159)
    print(jax.make_jaxpr(genjax.simulate(model))(key, ()))

That's quite a lot of new code! What :code:`genjax.simulate` is doing "under-the-hood" is expanding :code:`trace` to support:

* sampling from the :code:`gen_fn` (keyword argument to :code:`trace`).
* updating JAX's PRNG key.
* recording the log probability density of the resultant sample.
* returning these pieces of data out.

This essential pattern is repeated for each :doc:`interface` method.

.. _Marco Cusumano-Towner's thesis: https://www.mct.dev/assets/mct-thesis.pdf
.. _Gen.jl: https://github.com/probcomp/Gen.jl

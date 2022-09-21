What is a generative function? 
==============================

A generative function is a computational object which supports a concise
set of interfaces designed to support customizable Bayesian inference 
(*programmable inference*).

Formally, generative functions are mathematical representation of probabilistic
models that are expressive enough to permit models with random structure,
including capturing notions of variable existence uncertainty. This is described in `Marco Cusumano-Towner's thesis`_.

  🧠 **(A bit of knowledge) Gen.jl** 🧠

  One useful reference implementation of these objects lies in `Gen.jl`_,
  an encoding of generative functions and inference in Julia. 

  In GenJAX, our implementation of these objects is akin to the 
  :code:`static modeling language` of `Gen.jl`_ - we rely upon JAX to provide us
  with a useful intermediate representation for programs that we operate on 
  using transformations.

Speaking of the interface above, if you'd like to jump right to reading about this, visit :doc:`interface`.

What do generative functions look like in GenJAX?
-------------------------------------------------

A generative function in GenJAX looks like a pure Python function,
roughly the subset of Python acceptable by JAX.

.. jupyter-execute::

    import jax
    import genjax

    @genjax.gen
    def model(key):
        key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
        return key, x

    print(model)

This decorator returns a :code:`BuiltinGenerativeFunction` - a generative function
which implements the :doc:`interface` by utilizing JAX's tracing and program
transformation abilities.

Let's study the JAX representation of the code for this object.

.. jupyter-execute::
    
    key = jax.random.PRNGKey(314159)
    jaxpr = jax.make_jaxpr(model)(key)
    print(jaxpr.pretty_print(use_color=False))

In this lowered form, we can see that our JAX representation has a call to something called :code:`trace`, with a few constant keyword arguments (including an :code:`addr` and a :code:`gen_fn` as constant metadata associated with the operation).

JAX doesn't natively know how to handle :code:`trace` - it's a primitive that we've defined to support the generative function interface semantics. We get to tell JAX how to interpret this operation in terms of operations that it understands.

The interfaces are defined on :code:`BuiltinGenerativeFunction` by implementing transformations which operate on this code representation to implement the semantics. Here's one interface - :code:`simulate` - whose semantics require that we return a representation of the execution (a :code:`Trace`) along with an updated :code:`PRNGKey`.

.. jupyter-execute::
    
    key = jax.random.PRNGKey(314159)
    jaxpr = jax.make_jaxpr(genjax.simulate(model))(key, ())
    print(jaxpr.pretty_print(use_color=False))

That's quite a lot of new code! What :code:`genjax.simulate` is doing "under-the-hood" is expanding the :code:`trace` primitive to support:

* recursively calling :code:`simulate` on :code:`gen_fn` (keyword argument to :code:`trace`).
* updating JAX's PRNG key.
* recording the log probability density of the resultant sample.
* returning these pieces of data out (in a :code:`Trace` instance)

All this is happening before runtime -- JAX's tracer stages out these transformations into the clean (:code:`trace` primitive free) :code:`Jaxpr` above.

This essential pattern is repeated for each :doc:`interface` method defined on :code:`BuiltinGenerativeFunction`. 

Other generative function "languages" in GenJAX support their own implementations of each of the interface methods. Importantly, a generative function *need not* introspect on the implementation of a callee's interface methods. As long as the implementation is JAX compatible (and the semantics are implemented correctly), generative functions can call or utilize (c.f. :doc:`combinators`) other generative functions in composable patterns.

.. _Marco Cusumano-Towner's thesis: https://www.mct.dev/assets/mct-thesis.pdf
.. _Gen.jl: https://github.com/probcomp/Gen.jl

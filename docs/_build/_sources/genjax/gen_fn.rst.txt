Anatomy of a generative function
================================

A generative function is a computational object which supports a concise
set of interfaces designed to support customizable Bayesian inference
(*programmable inference*) and differentiable programming.

Formally, generative functions are mathematical representation of probabilistic
models that are expressive enough to permit models with random structure,
including capturing notions of variable existence uncertainty.
The framework is fully described in `Marco Cusumano-Towner's thesis`_.

.. admonition:: 🧠 **(A bit of knowledge) Gen.jl** 🧠

  The canonical reference implementation of these objects lies in `Gen.jl`_,
  an encoding of generative functions and inference in Julia.

  In GenJAX, our implementation of these objects is akin to the
  :code:`static modeling language` of `Gen.jl`_ - we rely upon JAX to provide us
  with a useful intermediate representation for programs that we operate on
  using transformations.

If you'd like to jump right to reading about the generative function interface, visit :doc:`interface`.

Distributions are generative functions
--------------------------------------

.. note::

   In this section, we won't cover the gradient interfaces. These are formally specified in
   :doc:`interface`, and discussed later in the :doc:`learning` and inference sections.

Let's start with simple examples of generative function -- classical distributions (objects like :code:`Normal`, or :code:`Uniform`).

Here's an implementation of one of these objects (GenJAX exposes :code:`Normal` as an instantiated alias of :code:`_Normal`)

::

  class _Normal(Distribution):
      def sample(self, key, mu, std, **kwargs):
          return mu + std * jax.random.normal(key, **kwargs)

      def logpdf(self, key, v, mu, std, **kwargs):
          z = (v - mu) / std
          return jnp.sum(
              -1.0
              * (jnp.square(jnp.abs(z)) + jnp.log(2.0 * pi))
              / (2 - jnp.log(std))
          )

      def __trace_type__(self, key, mu, std, **kwargs):
          shape = kwargs.get("shape", ())
          return Reals(shape)

  Normal = _Normal()

This object should roughly match intuition of what a distribution should provide:

1. (:code:`sample`) The ability to sample a value in the target space given access
   to a bit of pseudo-randomness.

2. (:code:`logpdf`) The ability to assess the log density of any sampled
   value from the target space, given access to parameters of
   the distribution (here: :code:`mu` and :code:`std`)

Now, let's examine :code:`simulate` - one of the generative function interface methods
whose semantics are described below:

.. admonition:: Simulate

   Given a generative function representing a normalized probability measure with choice map density :math:`t \sim P(\cdot; x)`, and return value :math:`r \sim P(\cdot; x, t)`:

   * Sample :math:`t` and :math:`r` from their associated probability measures (defined above).

   * Return a trace (holding :math:`t`, :math:`r`, the score of the sample :math:`P(t; x)`, and the arguments to the call).

   .. hint::

      C.f. `simulate in the Gen.jl docs`_.

.. _simulate in the Gen.jl docs: https://www.gen.dev/stable/ref/gfi/#Gen.simulate

.. admonition:: Working with log probabilities

   The formal description of these interfaces does not utilize logspace representations of density evaluations - but computationally, it's highly beneficial to work in logspace - especially for numerical stability.

   Throughout the documentation, we'll be using logspace representations of density evaluations.

By studying the specification for :code:`simulate`, we observe that generative functions are characterized by two types: the type of :code:`t` (the choice map type) and the type of :code:`r` (the return type). Let's refer to these types using uppercase to distinguish them from their value elements. Given this information, let's denote generative function objects as :math:`\mathbb{G}[T, R]`

To implement distributions as generative functions, we get to choose :math:`T` and :math:`R`. A natural choice would be to choose that these are the same type, and this type is the type of the target space (so, e.g., for :code:`Normal`, :math:`T` and :math:`R` would be :math:`\mathbb{R}`).

Now, :code:`simulate` implores us to sample from :math:`T`, compute :math:`R`, score :math:`T` - and then return a :code:`Trace` representation of this process.

Here's an implementation of :code:`simulate` which utilizes the interfaces on distribution objects that we described above:

::

  def simulate(self, key, args, **kwargs):
      key, sub_key = jax.random.split(key)
      v = self.sample(sub_key, *args, **kwargs)
      key, sub_key = jax.random.split(key)
      score = self.logpdf(sub_key, v, *args)
      tr = DistributionTrace(self, args, ValueChoiceMap(v), score)
      return key, tr

A few things of note: we have a concrete :code:`DistributionTrace` object which allows us to record the information which :code:`simulate` asks from us (specifically, the log score, the sampled value, and the return value - here the same as the sampled value).

In addition (and not covered explicitly by the formal interface), we pass through a :code:`jax.random.PRNGKey` - this is necessitated by JAX's programming model, but does not imply any serious collisions with the formal description.

Let's study :code:`importance` next - this example will provide a simple model of this interface, but there are more complexities lurking under the surface. We'll defer a longer discussion to the :doc:`interface` page.

.. admonition:: Importance

   Given a generative function representing a normalized probability measure with choice map density :math:`t \sim P(\cdot; x)`, and return value :math:`r \sim P(\cdot; x, t)`, as well as a set of constraints for random choices :math:`u`:

   * Sample :math:`t \sim Q(\cdot; u, x)` and :math:`r \sim Q(\cdot; x, t)` from proposals which are absolutely continuous with respect to the unnormalized conditional density on unconstrained random choices from :math:`P`.

   * Compute an importance weight :math:`\log \frac{P(t, r; x)}{Q(t; u, x)Q(r; x, t)}`.

   * Return the trace (which is consistent with the constraints :math:`u`) and the importance weight.

   .. hint::

      C.f. `generate in the Gen.jl docs`_.

.. _generate in the Gen.jl docs: https://www.gen.dev/stable/ref/gfi/#Gen.generate

This may already seem confusing - how do we provide the proposals :math:`Q`, where do those come from? What does it mean for a trace to be consistent with constraints :math:`u`?

Let's go in reverse order - a trace is consistent with constraints :math:`u` if, for all random choices which :math:`u` provides values for, the trace takes those values at those choices. This is Gen's version of conditioning - and just like in normal Bayesian mechanics, conditioning transforms a normalized probability measure into an unnormalized one. But importance sampling provides a mechanism to compute estimates of the normalization constant (that's what the importance weight is). That's what this interface is encapsulating.

Fortunately for us, distributions only expose a single random choice! So when we condition on constraints, we get a normalized density (or log density, in our case) evaluation. Thus, in our implementation of :code:`importance` - we can ignore :math:`Q`, at least for now.

Function-like generative functions in GenJAX
--------------------------------------------

GenJAX also exposes a modeling language which allows programmatic construction of generative functions based on pure Python functions (pure meaning: the subset of Python acceptable by JAX).

.. jupyter-execute::

    import jax
    import genjax

    @genjax.gen
    def model():
        x = genjax.trace("x", genjax.Normal)(0.0, 1.0)
        return x

    print(model)

This decorator returns a :code:`BuiltinGenerativeFunction` - a generative function
which implements the :doc:`interface` by utilizing JAX's tracing and program
transformation abilities.

Note, however, that we'll make recursive usage of the interface - here, we're calling distributions-as-generative-functions in our code! This is just one of the compositional benefits of constructing composite generative function objects using pieces that implement the interface.

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

Other generative function "languages" in GenJAX support their own implementations of each of the interface methods. Importantly, a generative function *need not* introspect on the implementation of a callee's interface methods. As long as the implementation is JAX compatible (and the semantics are implemented correctly), generative functions can call or utilize (c.f. :doc:`combinators/combinators`) other generative functions in composable patterns.

.. _Marco Cusumano-Towner's thesis: https://www.mct.dev/assets/mct-thesis.pdf
.. _Gen.jl: https://github.com/probcomp/Gen.jl

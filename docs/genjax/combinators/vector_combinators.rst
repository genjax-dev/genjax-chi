Vector combinators
==================

Several combinators transform the internal choice map space of a generative function by
taking advantage of JAX's broadcasting functionality and semantics. In this section, we'll
describe common usage patterns (especially concerning :code:`ChoiceMap` constraints) when 
utilizing these combinators.

.. attention::

   The content in this section describes a number of key differences between 
   :code:`Gen.jl` and GenJAX. For more information about other differences, examine
   the "diffs" page :doc:`../diff_jl`.

Example: Particle filtering with :code:`Unfold`
-----------------------------------------------

Let's take the examples above and study their usage in an inference algorithm
of moderate complexity: particle filtering.

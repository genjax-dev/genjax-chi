Diffing with Gen.jl
===================

:code:`GenJAX` is descended and inherits concepts and reference implementations from `Gen.jl`_ - but there are a few differences 
that mostly stem from JAX's underlying array programming model. In this section, we describe several of these differences and try to highlight
workarounds or discuss the reason for the discrepancy.

Turing universality
-------------------

`Gen.jl`_ is Turing universal - it can encode any computable distribution, including those
expressed by forms of unbounded recursion.

:code:`GenJAX` is not -- because JAX does not feature mechanisms for dynamic shape allocations.
While JAX does feature primitives for unbounded recursion, to support Gen's interfaces we also need
the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's
requirements of knowing the static shape of everything.

However, :code:`GenJAX` supports generative function combinators with bounded recursion / unfold chain length.
Ahead of time, these combinators can be directed to pre-allocate arrays with enough size to handle recursion/looping
within the bounds that the programmer sets. If these bounds are exceeded, a Python runtime error will be thrown (both on
and off JAX device). In practice, this means that some performance engineering (space vs. expressivity) is required of
the programmer.

To JIT or not to JIT
--------------------

`Gen.jl`_ is written in Julia, which automatically JITs everything. :code:`GenJAX`, by virtue of being
constructed on top of JAX, allows us to JIT JAX compatible code - but the JIT process is user directed.
Thus, the idioms that are used to express inference code has a small diff compared to `Gen.jl`_ - inference
algorithms normally accept some set of static arguments (like generative function instances) and return
functions which can be jitted by the user.

.. _Gen.jl: https://github.com/probcomp/Gen.jl

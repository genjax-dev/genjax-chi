Diffing with Gen.jl
===================

:code:`GenJAX` is descended and mostly inherits concepts and reference implementations from :code:`Gen.jl` - but there are a few differences 
that mostly stem from JAX's underlying array programming model. In this section, we describe several of these differences and try to highlight
workarounds or discuss the reason for the discrepancy.

Turing universality
-------------------

:code:`Gen.jl` is Turing universal - it can encode any computable distribution, including those
expressed by forms of unbounded recursion.

:code:`GenJAX` is not -- because JAX does not feature mechanisms for dynamic shape allocations.
While JAX does feature primitives for unbounded recursion, to support Gen's interfaces we also need
the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's
requirements of knowing the static shape of everything.

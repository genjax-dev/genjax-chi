# Diffing with Gen.jl

`GenJAX` is descended and inherits concepts and reference implementations from [`Gen.jl`][gen.jl] - but there are a few differences that mostly stem from JAX's underlying array programming model. In this section, we describe several of these differences and try to highlight workarounds or discuss the reason for the discrepancy.

## Turing universality

[`Gen.jl`][gen.jl] is Turing universal - it can encode any computable distribution, including those
expressed by forms of unbounded recursion.

It is a bit ambiguous whether or not `GenJAX` falls in this category: JAX does not feature mechanisms for dynamic shape allocations, but it does feature mechanisms for unbounded recursion.

The former provides a technical barrier to implementing Gen's trace machinery. While JAX allows for unbounded recursion, to support Gen's interfaces we also need the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's requirements of knowing the static shape of everything.

However, `GenJAX` supports generative function combinators with bounded recursion / unfold chain length.
Ahead of time, these combinators can be directed to pre-allocate arrays with enough size to handle recursion/looping
within the bounds that the programmer sets. If these bounds are exceeded, a Python runtime error will be thrown (both on
and off JAX device).

In practice, this means that some performance engineering (space vs. expressivity) is required of the programmer. It's certainly feasible to express bounded recursive computations which terminate with probability 1 - but you'll need to ahead of time allocate space for it.

## Mutation

Just like JAX, GenJAX disallows mutation - expressing a mutation to an array must be done through special interfaces, and those interfaces return full copies. There are special circumstances where these interfaces will be performed in place.

## To JIT or not to JIT

[`Gen.jl`][gen.jl] is written in Julia, which automatically JITs everything. `GenJAX`, by virtue of being constructed on top of JAX, allows us to JIT JAX compatible code - but the JIT process is user directed. Thus, the idioms that are used to express inference code has a small diff compared to [`Gen.jl`][gen.jl] - inference algorithms normally accept some set of static arguments (like generative function instances) and return functions which close over these arguments, and can be jitted by the user. You can read a bit more about this in :doc:`inference/best_practices`.

In general, it's productive to enclose as much of a computation as possible in a :code:`jax.jit` block. This can sometimes lead to long trace times. If trace times are ballooning, a common source is explicit for-loops (with known bounds, else JAX will complain). In these cases, you might look at [Advice on speeding up compilation time][jax speeding up compilation].

[gen.jl]: https://github.com/probcomp/Gen.jl
[jax speeding up compilation]: https://github.com/google/jax/discussions/3732

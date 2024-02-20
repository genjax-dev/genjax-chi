# Comparisons with Gen.jl

GenJAX implements concepts from Gen, and implements several inference algorithms with reference to implementations from [`Gen.jl`][gen.jl]. In general, you should find that the programming patterns and interface idioms should match closely with `Gen.jl`.

However, there are a few necessary design deviations between `genjax` and `Gen.jl` that stem from restrictions arising from JAX's compilation model. In this section, we describe several of these differences and try to highlight workarounds or discuss the reason for the discrepancy.

## Turing universality

[`Gen.jl`][gen.jl] is Turing universal - it can encode any computable distribution, including those
expressed by forms of unbounded recursion.

Arguing from a practical perspective, `genjax` also falls into this category, but several things are harder to encode for GPUs(1). Additionally, JAX does not feature mechanisms for dynamic shape allocations, but it does feature mechanisms for unbounded recursion.
{ .annotate }

1. We expect that GPU/TPU deployment to be the dominant usage pattern for `genjax` - and defer optimized CPU deployment to other implementations of Gen.

Lack of dynamic allocations provides a technical barrier to implementing Gen's trace machinery for generative functions which feature recursive calls to other generative functions. While JAX allows for unbounded recursion, to generally support recording trace data - we also need the ability to dynamically allocate choice data. This requirement is currently at tension with XLA's requirements of knowing the static shape of everything. Nonetheless, one might imagine pre-allocating large arrays - passing them into `jax.lax.while_op` implementations of certain types of recursion, etc. Painful and impractical - yes, but theoretical possible.

`genjax` supports generative function combinators with bounded recursion / unfold chain length.
Ahead of time, these combinators can be directed to pre-allocate arrays with enough size to handle recursion/looping
within the bounds that the programmer sets. If these bounds are exceeded, a Python runtime error will be thrown (both on
and off JAX device).

In practice, this means that some performance engineering (space vs. expressivity) is required of the programmer. It's certainly feasible to express bounded recursive computations which terminate with probability 1 - but you'll need to ahead of time allocate space for it.

## Mutation

Just like JAX, GenJAX disallows mutation - expressing a mutating operation on an array must be done through special JAX interfaces. Outside of JIT compilation, those interfaces often fully copy array data. Inside of JIT compilation, there are special circumstances where these operations will be performed in place.

## To JIT or not to JIT

[`Gen.jl`][gen.jl] is written in Julia, which automatically JITs everything. `genjax`, by virtue of being constructed on top of JAX, allows us to JIT JAX compatible code - but the JIT process is user directed. Thus, the idioms that are used to express and optimize inference code are necessarily different compared to [`Gen.jl`][gen.jl]. In the inference standard library, you'll typically find algorithms implemented as dataclasses which inherit (and implement) the `jax.Pytree` interfaces. Implementing these interfaces allow usage of inference dataclasses and methods in jittable code - and, as a bonus, allow us to be specific about trace vs. runtime known values.

In general, it's productive to enclose as much of a computation as possible in a `jax.jit` block. This can sometimes lead to long trace times. If trace times are ballooning, a common source is explicit for-loops (with known bounds, else JAX will complain). In these cases, you might look at [Advice on speeding up compilation time][jax speeding up compilation]. We've taken care to optimize (by e.g. using XLA primitives) the code which we expose from GenJAX - but if you find something out of the ordinary, file an issue!

[gen.jl]: https://github.com/probcomp/Gen.jl
[jax speeding up compilation]: https://github.com/google/jax/discussions/3732

# GenProx

This is a Python implementation (on top of JAX + GenJAX's Gen interfaces) of `GenProx` - a set of PPL concepts and types for programming with approximate densities and pseudo-marginalization.

The original source code was developed by Alex Lew in Julia here: https://github.com/probcomp/GenProx.jl.

A formal description of elements of this work can be found in Alex Lew's work on [Recursive auxiliary-variable inference](https://arxiv.org/abs/2203.02836) (will update with a formal citation).

<div align="center">
<b><i>
This is a WIP implementation of WIP research code, expect 🔪 edges, as always.
</i></b>
</div>

## The composable SMC mini-language

I had to modify the composable SMC (`composable_smc.py`) DSL to support compilation to XLA idioms which prevent code blowup (and long compile times). The primary concern here is using an implementation which JAX will fully inline (the original implementation was this way).

To handle this issue, this implementation is organized around _function types_ which return targets and combinators which operate on ingredients to transform those function types (in the code, `final_target` and `step_target`).

There are two classes of `SMC` ingredient:

`SMCRoot` instances are those SMC ingredients whose "target" generating function has a nullary signature: `() -> Target`, and whose "number of particles" function `num_particles` has nullary signature: `() -> int`. These are things like `SMCInit`, which takes an initial proposal, an initial target, and a static number of particles.

`SMCPropagator` instances are those SMC ingredients whose "target" generating functions require data (not compile-time known, but compile-time shape known) and a `Target` to create a new `Target` e.g. `(Target, Tuple, ChoiceMap) -> Target` for `SMCExtend` - which wants access to the previous target, new arguments, and new constraints to create a new target.

`SMCCombinator` instances are those SMC ingredients which accept combinations of `SMCRoot` and `SMCPropagator` instances to implement larger algorithm patterns. These combinator instances typically utilize XLA-specific control flow primitives which prevent compile-time code blow up. One useful example is `SMCChain` - this implements a scan-like pattern, for example:

```
prox.SMCChain(
  prox.SMCExtend(some_proposal),
  args=(some_array_args, vectorized_choice_map),
  chain_length = 50
)
```

where `some_array` has `0-dim` size `50` and `vectorized_choice_map` is a `Pytree` whose leaves all have size `50`.

`SMCChain` transforms a `SMCPropagator` into a new `SMCPropagator` which efficiently implements multiple SMC algorithm steps using `jax.lax.scan`.

Another combinator example is `SMCCompose` - which accepts a sequence of `SMCPropagator` instances and composes them.

# Combinators: structured patterns of composition

While the programmatic `StaticGenerativeFunction` language is powerful, its restrictions can be limiting. Combinators are a way to express common patterns of composition in a more concise way, and to gain access to effects which are common in JAX (like `jax.vmap`) for generative computations.

Each of the combinators below is implemented as a decorator. `GenerativeFunction` instances make each combinator available as a method with the same name.

## `vmap`-like Combinators

::: genjax.vmap
::: genjax.repeat

## `scan`-like Combinators

::: genjax.scan

## Control Flow Combinators

::: genjax.or_else
::: genjax.switch

## Various Transformations

::: genjax.map_addresses
::: genjax.dimap
::: genjax.map
::: genjax.contramap

## The Rest

::: genjax.mask
::: genjax.mix

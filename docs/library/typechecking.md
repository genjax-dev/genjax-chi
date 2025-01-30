# Enforcing type constraints

GenJAX uses [`beartype`](https://github.com/beartype/beartype) to perform type checking _during JAX tracing / compile time_. This means that `beartype`, normally a fast _runtime_ type checker, operates _at JAX tracing time_ to ensure that the arguments and return values are correct, with zero runtime cost.

###  Generative interface types

::: genjax.core.Arguments
::: genjax.core.Score
::: genjax.core.Weight
::: genjax.core.Retdiff
::: genjax.core.Argdiffs

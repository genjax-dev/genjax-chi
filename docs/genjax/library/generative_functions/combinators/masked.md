# Masked combinator

The `MaskedCombinator` exposes a useful utility for structuring generative computations: the ability to dynamically hide parts of the generative computation (in a safe way).

To support this functionality, `MaskedCombinator` utilizes `genjax.Mask` in the implementation of its generative function interface methods.

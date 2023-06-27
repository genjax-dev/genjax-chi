# Unfold combinator

GenJAX's `UnfoldCombinator` is a combinator which implements a scan-like pattern of control flow by utilizing `jax.lax.scan`.

::: genjax.generative_functions.combinators.UnfoldCombinator
    options:
      show_root_heading: true
      members:
        - new

## Choice maps for `Unfold`

`Unfold` produces `VectorChoiceMap` instances (a type of choice map shared with `MapCombinator`).

To utilize `importance`, `update`, or `assess` with `Unfold`, it suffices to provide either a `VectorChoiceMap` for constraints, or an `IndexChoiceMap`. Both of these choice maps are documented below (documentation is mirrored at `MapCombinator`).

::: genjax.generative_functions.combinators.VectorChoiceMap
    options:
      show_root_heading: true
      members:
        - new

::: genjax.generative_functions.combinators.IndexChoiceMap
    options:
      show_root_heading: true
      members:
        - new

## Selections for `VectorChoiceMap`

> (This section is also mirrored for `MapCombinator`)

To select from `VectorChoiceMap`, both `VectorSelection` and `IndexSelection` can be used. `VectorSelection`



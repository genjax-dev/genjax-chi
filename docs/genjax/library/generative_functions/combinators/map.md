# Map combinator

GenJAX's `MapCombinator` is a combinator which exposes vectorization to the input arguments of generative functions.

::: genjax.generative_functions.combinators.MapCombinator
    options:
      show_root_heading: true
      members:
        - new


## Choice maps for `Map`

> (This section is also mirrored for `UnfoldCombinator`)

`Map` produces `VectorChoiceMap` instances (a type of choice map shared with `UnfoldCombinator`).

To utilize `importance`, `update`, or `assess` with `Map`, it suffices to provide either a `VectorChoiceMap` for constraints, or an `IndexChoiceMap`. 

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

> (This section is also mirrored for `UnfoldCombinator`)

To `filter` from `VectorChoiceMap`, or `project` from `MapTrace` both `HierarchicalSelection` and `IndexSelection` can be used.

::: genjax.generative_functions.combinators.MapTrace
    options:
      show_root_heading: true
      members:
        - project

::: genjax.generative_functions.combinators.VectorChoiceMap
    options:
      show_root_heading: true
      members:
        - filter

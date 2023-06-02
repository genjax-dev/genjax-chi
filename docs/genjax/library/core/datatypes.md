# Core datatypes

GenJAX features a set of core abstract datatypes which build on JAX's `Pytree` interface. These datatypes are used as an abstract base mixin (especially GenJAX's `Pytree` utility abstract base class) for basically all of the dataclasses in GenJAX.

## Pytree

::: genjax.core.Pytree
    options:
      members: 
        - flatten
        - unflatten
        - slice
        - stack
        - unstack

## Abstract base classes which extend `Pytree`

::: genjax.core.Tree
    options:
      members: 
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

::: genjax.core.Leaf
    options:
      members: 
        - get_leaf_value
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

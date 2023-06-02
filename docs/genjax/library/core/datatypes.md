# Core datatypes

GenJAX features a set of core abstract datatypes which build on JAX's `Pytree` interface. These datatypes are used as an abstract base mixin (especially `Pytree`) for basically all of the dataclasses in GenJAX.

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

### Tree

The `Tree` class is used to define abstract classes for tree-shaped datatypes. These classes are used to implement trace, choice map, and selection types. `Tree` mixes in `Pytree` automatically.

One should think of `Tree` as providing a convenient base for many of the generative datatypes.

::: genjax.core.Tree
    options:
      members: 
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

### Leaf

A `Leaf` is a `Tree` without any internal subtrees. `Leaf` is a convenient base for generative datatypes which don't keep reference to other `Tree` instances - things like `ValueChoiceMap` (whose only choice value is a single value, not a dictionary or other tree-like object).

`Leaf` extends `Tree` with a special extension method `get_leaf_value`.

::: genjax.core.Leaf
    options:
      members: 
        - get_leaf_value
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

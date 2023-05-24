# Core datatypes

GenJAX features a set of core abstract datatypes which build on JAX's `Pytree` interface. These datatypes are used as an abstract base mixin (especially `Pytree`) for basically all of the dataclasses in GenJAX.

::: genjax.core.Pytree
    options:
      members: 
        - flatten
        - unflatten

## Trees

The `Pytree` class is used to define abstract classes for tree-shaped datatypes. These classes are used to implement trace, choice map, and selection types.

::: genjax.core.Tree
    options:
      members: 
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

### Leaf

A `Leaf` is a `Tree` without any internal subtrees.

::: genjax.core.Leaf
    options:
      members: 
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

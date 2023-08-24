# Core datatypes

GenJAX exposes a set of core abstract classes which build on JAX's `Pytree` interface. These datatypes are used as abstract base mixins for many of the key dataclasses in GenJAX.
    
::: genjax.core.Pytree
    options:
      members: 
        - flatten
        - unflatten
        - slice
        - stack
        - unstack

::: genjax.core.AddressTree
    options:
      members: 
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

::: genjax.core.AddressLeaf
    options:
      members: 
        - get_leaf_value
        - has_subtree
        - get_subtree
        - get_subtrees_shallow

# Functional types

GenJAX provides a set of extension `Pytree` types to enable modeling idioms which require runtime uncertainty - including models with switching (including address set inhomogeneity), and state space models with dynamic length. To encode runtime uncertainty, we utilize JAX encodings of functional option and sum types. We describe these types below, as well as their usage.

## (Option types) The masking system

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true

## (Sum types) Tagged unions

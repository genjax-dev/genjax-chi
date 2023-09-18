# Specialization, value universes, and functional types

**GenJAX** consists of two ingredients: Gen & JAX. The semantics of Gen are defined independently of the concerns of any particular computational substrate used in the implementation of Gen - it is our responsibility as the implementers to prevent these concerns from "leaking upwards" to modify the semantics of Gen and its interface specifications.

JAX is a unique substrate. While not formally modelled, it's appropriate to think of JAX as separating computation into two phases: a _statics_ phase (which occurs at JAX tracing time) and a _runtime_ phase (which occurs when a computation written in JAX is actually deployed to a physical device). JAX has different rules for handling values depending on which phase we are in - e.g. JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

This provides the implementers of Gen a large deal of freedom in constructing code which, when traced, produces specialized code _depending on static information_. At the same time, we must be very careful to encode Gen's interfaces while respecting JAX's rules which govern how static / runtime values can be used.

In this section, we will give an overview of several patterns used in GenJAX which navigate this trade off. In general, users should not be required to be aware of the details in this section - but it may be useful for advanced users who are concerned about performance properties of their code (and optimization opportunities) or advanced users who are seeking to implement new generative function languages in JAX proper.

## Specialized choice map representations

## Functional types

GenJAX provides a set of extension `Pytree` types to enable modeling idioms which require runtime uncertainty - including models with switching (including address set inhomogeneity), and state space models with dynamic length. To encode runtime uncertainty, we utilize JAX encodings of functional option and sum types. We describe these types below, as well as their usage.

### (Option types) The masking system

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true
        members:
          - match
          - unmask

### (Sum types) Tagged unions

Like option types, sum types allow representing a form of type uncertainty which can be useful when working within the restricted `jax.lax` control flow model.

::: genjax.core.TaggedUnion
    options:
        show_root_heading: true
        members:
          - match

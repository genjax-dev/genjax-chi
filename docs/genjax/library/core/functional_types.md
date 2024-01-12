# Dynamism and masking
!!! note "Navigating the static/dynamic trade off"

    This page provides an overview of several patterns used in GenJAX which navigate this _statics vs. dynamics_ trade off. Depending on the modeling and inference application, users may be required to confront this trade off. This page provides documentation on using _masking_, one technique which allows us to push static decisions to runtime, via the insertion of `jax.lax.cond` statements.

**GenJAX** consists of two ingredients: Gen & JAX. The semantics of Gen are defined independently of the concerns of any particular computational substrate used in the implementation of Gen - that being said, JAX (and XLA through JAX) is a unique substrate, offering high performance, with a unique set of restrictions.

While not yet formally modelled, it's appropriate to think of JAX as separating computation into two phases:

* The _statics_ phase (which occurs at JAX tracing time).
* The _runtime_ phase (which occurs when a computation written in JAX is actually deployed and executed on a physical device).

JAX has different rules for handling values depending on which phase we are in e.g. JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

We take advantage of JAX's tracing to construct code which, when traced, produces specialized code _depending on static information_. At the same time, we must be very careful to encode Gen's interfaces while respecting JAX's rules which govern how static / runtime values can be used.

### The masking system

The most primitive way to encode _runtime uncertainty_ about a piece of data is to attach a `Bool` to it, which indicates whether the data is "on" or "off".

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true
        members:
          - match
          - unmask

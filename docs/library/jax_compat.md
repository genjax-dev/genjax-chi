# Our approach to JAX compatibility

JAX natively works with arrays and lists of arrays, but, for our own convenience, JAX exposes an interface called Pytrees which allow users to define Python classes and use them in JAX computations ... so long as the class tells JAX how to break its instances down into lists of arrays. You can read more about the system [`here`](https://jax.readthedocs.io/en/latest/pytrees.html).

All of the datatypes found in GenJAX are compatible with JAX through the Pytree interface: GenJAX provides an abstract class called `Pytree` which automates the implementation of the `flatten` / `unflatten` methods for a class.

GenJAX's `Pytree` inherits from [`penzai.Struct`](https://penzai.readthedocs.io/en/stable/_autosummary/leaf/penzai.core.struct.Struct.html), which provides some convenient methods to annotate what data should be part of the `Pytree` _type_ (static fields, won't be broken down into a JAX array) and what data should be considered dynamic.

::: genjax.core.Pytree
    options:
      members:
        - dataclass
        - static
        - field

There are several useful utility `Pytree` classes which you may find in your GenJAX journey. `genjax.core.Const` represents a _static_ (known at JAX tracing time) value. `genjax.core.Closure` provides a `Pytree` representation of a Python closure.

::: genjax.core.Const

::: genjax.core.Closure

## Dynamism in JAX: masks and sum types

The semantics of Gen are defined independently of any particular computational substrate or implementation - but JAX (and XLA through JAX) is a unique substrate, offering high performance, the ability to transformation code ahead-of-time via program transformations, and ... _a rather unique set of restrictions_.

### JAX is a two-phase system

It's appropriate to think of JAX as separating computation into two phases:

* The _statics_ phase (which occurs at JAX tracing / transformation time).
* The _runtime_ phase (which occurs when a computation written in JAX is actually deployed via XLA and executed on a physical device somewhere in the world).


JAX has different rules for handling values depending on which phase we are in.

For instance, JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

In GenJAX, we take advantage of JAX's tracing to construct code which, when traced, produces specialized code _depending on static information_. At the same time, we are careful to encode Gen's interfaces to respect JAX's rules which govern how static / runtime values can be used.

The most primitive way to encode _runtime uncertainty_ about a piece of data is to attach a `bool` to it, which indicates whether the data is "on" or "off".

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true
        members:
          - unmask
          - match

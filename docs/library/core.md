# Journey to the center of `genjax.core`

GenJAX consists of a _core_ set of functionality and datatypes, used to enforce consistent interfaces, build up generative functions, and ensure JAX compatibility. This page describes the set of core datatypes in GenJAX, including the core JAX compatibility datatypes ([`Pytree`][genjax.core.Pytree]), and the key Gen generative datatypes ([`GenerativeFunction`][genjax.core.GenerativeFunction], [`Trace`][genjax.core.Trace], [`Sample`][genjax.core.Sample], [`Constraint`][genjax.core.Constraint], and [`UpdateSpec`][genjax.core.UpdateSpec]).

This page also describes GenJAX's approach to [full JAX compatibility](core.md#we-jax-everything-is-jax-compatible-by-default).

## Generative types

The main computational objects in Gen are _generative functions_. These objects support an abstract interface of methods and associated types. The interface is designed to allow the implementations of Bayesian inference algorithms to abstract over the implementation of common subroutines (like computing importance weights, or accept-reject ratios).

::: genjax.core.GenerativeFunction
    options:
      members:
        - simulate
        - importance
        - assess
        - update

Traces are data structures which record (execution and inference) data about the invocation of generative functions. Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations. Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
    options:
      members:
        - get_args
        - get_retval
        - get_gen_fn
        - get_sample
        - get_score

::: genjax.core.Sample
::: genjax.core.Constraint
::: genjax.core.UpdateSpec

## Frequently used types of (`Sample`, `Constraint`, `UpdateSpec`)

## We ♥ JAX: everything is JAX compatible by default

GenJAX exposes a set of core abstract classes which build on JAX's `Pytree` interface. These datatypes are used as abstract base mixins for many of the key dataclasses in GenJAX.

::: genjax.core.Pytree
    options:
      members:
        - dataclass
        - static
        - field
        - __getitem__

### Dynamism and masking

It's in the name! GenJAX consists of two ingredients: Gen & JAX. The semantics of Gen are defined independently of any particular computational substrate or implementation - but JAX (and XLA through JAX) is a unique substrate, offering high performance, the ability to transformation code ahead-of-time via program transformations, and ... _a rather unique set of restrictions_.

While not yet formally modelled, it's appropriate to think of JAX as separating computation into two phases:

* The _statics_ phase (which occurs at JAX tracing / transformation time).
* The _runtime_ phase (which occurs when a computation written in JAX is actually deployed via XLA and executed on a physical device somewhere in the world).

JAX has different rules for handling values depending on which phase we are in e.g. JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

We take advantage of JAX's tracing to construct code which, when traced, produces specialized code _depending on static information_. At the same time, we must be very careful to encode Gen's interfaces while respecting JAX's rules which govern how static / runtime values can be used.

#### The masking system

The most primitive way to encode _runtime uncertainty_ about a piece of data is to attach a `Bool` to it, which indicates whether the data is "on" or "off".

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true
        members:
          - match
          - unmask

## Static typing with `genjax.typing` a.k.a 🐻`beartype`🐻

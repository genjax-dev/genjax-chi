# Journey to the center of `genjax.core`


This page describes the set of core concepts and datatypes in GenJAX, including Gen's generative datatypes and concepts ([`GenerativeFunction`][genjax.core.GenerativeFunction], [`Trace`][genjax.core.Trace], [`Sample`][genjax.core.Sample], [`Constraint`][genjax.core.Constraint], and [`EditRequest`][genjax.core.EditRequest]), the core JAX compatibility datatypes ([`Pytree`][genjax.core.Pytree], [`Const`][genjax.core.Const], and [`Closure`][genjax.core.Closure]), as well as functionally inspired `Pytree` extensions ([`Mask`][genjax.core.Mask]), and GenJAX's approach to "static" (JAX tracing time) typechecking.

::: genjax.core.GenerativeFunction

Traces are data structures which record (execution and inference) data about the invocation of generative functions. Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations. Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
::: genjax.core.Sample
::: genjax.core.EditRequest
::: genjax.core.Constraint

## Generative functions with addressed random choices

Generative functions will often include _addressed_ random choices. These are random choices which are given a name via an addressing syntax, and can be accessed by name via extended interfaces on the `Sample` type which supports the addressing.

The standard `Sample` type for this type of generative function is the `ChoiceMap` type.

::: genjax.core.ChoiceMap
::: genjax.core.Selection

## JAX compatible data via `Pytree`

JAX natively works with arrays, and with instances of Python classes which can be broken down into lists of arrays. JAX's [`Pytree`](https://jax.readthedocs.io/en/latest/pytrees.html) system provides a way to register a class with methods that can break instances of the class down into a list of arrays (canonically referred to as _flattening_), and build an instance back up given a list of arrays (canonically referred to as _unflattening_).

GenJAX provides an abstract class called `Pytree` which automates the implementation of the `flatten` / `unflatten` methods for a class. GenJAX's `Pytree` inherits from [`penzai.Struct`](https://penzai.readthedocs.io/en/stable/_autosummary/leaf/penzai.core.struct.Struct.html), to support pretty printing, and some convenient methods to annotate what data should be part of the `Pytree` _type_ (static fields, won't be broken down into a JAX array) and what data should be considered dynamic.

::: genjax.core.Pytree
    options:
      members:
        - dataclass
        - static
        - field

::: genjax.core.Const

::: genjax.core.Closure
## Dynamism in JAX: masks and sum types

The semantics of Gen are defined independently of any particular computational substrate or implementation - but JAX (and XLA through JAX) is a unique substrate, offering high performance, the ability to transformation code ahead-of-time via program transformations, and ... _a rather unique set of restrictions_.

### JAX is a two-phase system

While not yet formally modelled, it's appropriate to think of JAX as separating computation into two phases:

* The _statics_ phase (which occurs at JAX tracing / transformation time).
* The _runtime_ phase (which occurs when a computation written in JAX is actually deployed via XLA and executed on a physical device somewhere in the world).


JAX has different rules for handling values depending on which phase we are in.

For instance, JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

In GenJAX, we take advantage of JAX's tracing to construct code which, when traced, produces specialized code _depending on static information_. At the same time, we are careful to encode Gen's interfaces to respect JAX's rules which govern how static / runtime values can be used.

The most primitive way to encode _runtime uncertainty_ about a piece of data is to attach a `Bool` to it, which indicates whether the data is "on" or "off".

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
    options:
        show_root_heading: true
        members:
          - unmask
          - match

## Static typing with `genjax.typing` a.k.a 🐻`beartype`🐻

GenJAX uses [`beartype`](https://github.com/beartype/beartype) to perform type checking _during JAX tracing / compile time_. This means that `beartype`, normally a fast _runtime_ type checker, operates _at JAX tracing time_ to ensure that the arguments and return values are correct, with zero runtime cost.

###  Generative interface types

::: genjax.core.Arguments
::: genjax.core.Score
::: genjax.core.Weight
::: genjax.core.Retdiff
::: genjax.core.Argdiffs

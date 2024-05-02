# Core datatypes

This page describes the set of core datatypes in GenJAX, including the core JAX compatibility layer datatypes (`Pytree`), and the key Gen generative datatypes (`GenerativeFunction`, `Trace`, `Choice` & `ChoiceMap`, and `Selection`).

!!! note "Key generative datatypes in Gen"

    This documentation page contains the type and interface documentation for the core generative datatypes used in Gen. The documentation on this page deals with the abstract base classes for these datatypes.

    **Any concrete (or specialized) implementor of these datatypes should be documented with the language which implements it.** Specific generative function languages are not documented here, although they may be used in example code fragments.


## The `Pytree` data layer

GenJAX exposes a set of core abstract classes which build on JAX's `Pytree` interface. These datatypes are used as abstract base mixins for many of the key dataclasses in GenJAX.

::: genjax.core.Pytree
    options:
      members:
        - flatten
        - unflatten
        - slice
        - stack
        - unstack

## Core generative datatypes

!!! tip "Generative functions, traces, choice types, and selections"

    The data types discussed below are critical to the design of Gen, and are the main data types that users can expect to interact with.

The main computational objects in Gen are _generative functions_. These objects support an abstract interface of methods and associated types. The interface is designed to allow the implementations of Bayesian inference algorithms to abstract over the implementation of common subroutines (like computing an importance weight, or an accept-reject ratio).

Below, we document the abstract base class `GenerativeFunction`, and illustrate example usage of the method interface (`simulate`, `importance`, `update`, and `assess`). Full descriptions of concrete generative function languages are described in their own documentation module (c.f. [Generative function language](../generative_functions/index.md)).

!!! info "Logspace for numerical stability"

    In Gen, all relevant inference quantities are given in logspace(1). Most implementations also use logspace, for the same reason. In discussing the math below, we'll often say "the score" or "an importance weight" and drop the $\log$ modifier as implicit.
    { .annotate }

    1. For more on numerical stability & log probabilities, see [Log probabilities](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/log_probabilities/).

::: genjax.core.GenerativeFunction
    options:
      members:
        - simulate
        - propose
        - importance
        - assess
        - update

## The `Trace` type

Traces are data structures which record (execution and inference) data about the invocation of generative functions.

Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations.

Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
    options:
      members:
        - get_gen_fn
        - get_retval
        - get_choices
        - get_score
        - project

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

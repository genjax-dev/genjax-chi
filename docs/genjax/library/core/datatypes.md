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

## JAX compatible generative functions

The interface definitions of generative functions may interact with JAX tracing machinery. GenJAX does not strictly impose this requirement on generative function implementations, but does provide a generative function class called `JAXGenerativeFunction` which denotes compatibility assumptions with JAX tracing.

Other generative function languages which utilize callee generative functions can enforce JAX compatibility by typechecking on `JAXGenerativeFunction`. See, for instance, the [generative function combinators](../generative_functions/combinators/index.md) which expect JAX compatible generative functions as callees.

::: genjax.core.JAXGenerativeFunction
    options:
      members:
      - unzip
      - choice_grad

## Traces

Traces are data structures which record (execution and inference) data about the invocation of generative functions.

Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations.

Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
    options:
      members:
        - get_gen_fn
        - get_retval
        - get_choice
        - get_score
        - project

## The `Choice` type

::: genjax.core.Choice
    options:
      members: False

::: genjax.core.EmptyChoice
    options:
      members: False

::: genjax.core.ChoiceValue
    options:
      members: False

::: genjax.core.ChoiceMap
    options:
      members:
        - filter
        - insert
        - replace

## The `Selection` type

::: genjax.core.Selection
    options:
      members:
        - complement

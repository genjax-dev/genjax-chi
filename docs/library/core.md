# Journey to the center of `genjax.core`


This page describes the set of core concepts and datatypes in GenJAX, including Gen's generative datatypes and concepts ([`GenerativeFunction`][genjax.core.GenerativeFunction], [`Trace`][genjax.core.Trace], [`ChoiceMap`][genjax.core.ChoiceMap], [`Constraint`][genjax.core.Constraint], and [`EditRequest`][genjax.core.EditRequest]), the core JAX compatibility datatypes ([`Pytree`][genjax.core.Pytree], [`Const`][genjax.core.Const], and [`Closure`][genjax.core.Closure]), as well as functionally inspired `Pytree` extensions ([`Mask`][genjax.core.Mask]), and GenJAX's approach to "static" (JAX tracing time) typechecking.

::: genjax.core.GenerativeFunction
    options:
        members:
            - simulate
            - assess
            - generate
            - edit
            - support

Traces are data structures which record data about the invocation of generative functions. Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations. Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
    options:
        members:
            - get_args
            - get_retval
            - get_gen_fn
            - get_choices
            - get_score
            - project
            - edit

## Generative functions with addressed random choices

Generative functions typically include _addresses_ to represent random choices. Samples are reified into a type of nested structure called a `ChoiceMap`. The type `ChoiceMap` allows the values of random choices to be accessed via indexing using the addresses.

::: genjax.core.ChoiceMap
::: genjax.core.Selection

## Inference interfaces and datatypes

### The `generate` interface

::: genjax.core.Constraint

### The `edit` interface

::: genjax.core.EditRequest

## Derived interfaces

The `edit` and `generate` interfaces are powerful, and form the basis for several types of automatic sampling, which are exposed via additional interfaces.

::: genjax.core.GenerativeFunction.importance

::: genjax.core.GenerativeFunction.update

::: genjax.core.Trace.update

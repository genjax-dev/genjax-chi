# Generative datatypes

## Generative functions

The main computational objects in Gen are _generative functions_. These objects support an abstract interface of methods and associated types. The interface is designed to allow inference layers to abstract over implementations.

Below, we document the abstract base class, and illustrate example usage using concrete implementors. Full descriptions of concrete generative function languages are described in their own documentation module.

!!! info "Logspace for numerical stability"

    In Gen, all relevant inference quantities are given in logspace(1). Most implementations also use logspace, for the same reason. In discussing the math below, we'll often say "the score" or "an importance weight" and drop the $\log$ modifier as implicit.
    { .annotate }

    1. For more on numerical stability & log probabilities, see [Log probabilities](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/log_probabilities/).

!!! note "Key generative datatypes in Gen"
    
    This documentation page contains the type and interface documentation for the primary generative datatypes used in Gen. The documentation on this page deals with the abstract base classes for these datatypes. 

    **Any concrete `GenerativeFunction` implementor should be documented with the language which implements it.** Thus, specific generative function languages are not documented here.


::: genjax.core.GenerativeFunction
    options:
      members: 
        - simulate
        - propose
        - importance
        - assess
        - update

The interface definitions of generative functions often interact with JAX tracing machinery. GenJAX does not strictly impose this requirement, but does provide a generative function subclass called `JAXGenerativeFunction` which provides default compatibility definitions for JAX tracing.

Other generative function languages which utilize callee generative functions can enforce JAX compatibility by typechecking on `JAXGenerativeFunction`.

::: genjax.core.JAXGenerativeFunction
    options:
      members: false

## Traces

Traces are data structures which record (execution and inference) data about the invocation of generative functions.

Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations.

Traces support a set of accessor method interfaces designed to provide convenient manipulation when handling traces in inference algorithms.

::: genjax.core.Trace
    options:
      members: 
        - get_gen_fn
        - get_retval
        - get_choices
        - get_score
        - strip
        - project

## Choice maps

::: genjax.core.ChoiceMap
    options:
      members: 
        - filter
        - insert
        - replace

## Selections

::: genjax.core.Selection
    options:
      members: 
        - complement

# Generative datatypes

!!! note "Key generative datatypes in Gen"
    
    This documentation page contains the type and interface documentation for the primary generative datatypes used in Gen. The documentation on this page deals with the abstract base classes for these datatypes. 

    **Any concrete implementor of these abstract classes should be documented with the language which implements it.**

## Generative functions

The main computational objects in Gen are _generative functions_. These objects support an abstract interface of methods and associated types. The interface is designed to allow inference layers to abstract over implementations.

Below, we document the abstract base class, and illustrate example usage using concrete implementors. Full descriptions of concrete generative function languages are described in their own documentation module.

::: genjax.core.GenerativeFunction
    options:
      members: 
        - simulate
        - propose
        - importance
        - assess
        - update

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

## Selections

::: genjax.core.Selection

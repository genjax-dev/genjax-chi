# Generative datatypes

!!! note "Key generative datatypes in Gen"
    
    This documentation page contains the type and interface documentation for the primary generative datatypes used in Gen. The documentation on this page deals with the abstract base classes for these datatypes. 

    **Any concrete implementor of these abstract classes should be documented with the language which implements it.**

## Generative functions

The main computational objects in Gen are _generative functions_. These objects support an abstract interface of methods and associated types which allow inference layers to abstract over the implementation of the interface.

Below, we document the base abstract class. Concrete generative function languages are described in their own documentation module.

::: genjax.core.GenerativeFunction
    options:
      members: 
        - simulate
        - importance
        - update
        - assess

## Traces

::: genjax.core.Trace
    options:
      members: 
        - get_gen_fn
        - get_retval
        - get_choices
        - get_score
        - strip

## Choice maps

::: genjax.core.ChoiceMap

## Selections

::: genjax.core.Selection

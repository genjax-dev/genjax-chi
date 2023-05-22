# Generative datatypes

## Generative functions

The main computational object of Gen is _the generative function_. These objects support a method and associated type interface which allows inference layers to abstract over the interface implementation.

Below, we document the base abstract class. Concrete generative function languages are described in their own documentation module.

::: genjax.core.GenerativeFunction
    options:
      members: 
        - simulate
        - importance
        - update
        - assess

## Traces

## Choice maps

## Selections

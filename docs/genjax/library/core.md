::: genjax._src.core
    options:
      show_root_heading: false

---

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

---

## Interpreters

JAX supports transformations of pure, numerical Python programs by staging out interpreters which evaluate [`Jaxpr`](https://jax.readthedocs.io/en/latest/jaxpr.html) representations of programs.

The `Core` module features interpreter infrastructure, and common transforms designed to facilitate certain types of transformations.

### Contextual interpreter

A common type of interpreter involves overloading desired primitives with context-specific behavior by inheriting from `Trace` and define the correct methods to process the primitives.

In this module, we provide an interpreter which mixes initial style (e.g. the Python program is immediately staged, and then an interpreter walks the `Jaxpr` representation) with custom `Trace` and `Tracer` overloads. 

This pattern supports a wide range of program transformations, and allows parametrization over the inner interpreter (e.g. forward evaluation, or CPS).

::: genjax._src.core.interpreters.context
    options:
      members: 
        - ContextualTracer
        - ContextualTrace

### `harvest` via contextual interpreter

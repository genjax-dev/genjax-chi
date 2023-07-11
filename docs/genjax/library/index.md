# Library reference

This is the API documentation for modules and symbols which are exposed publicly from `genjax`. 

The `genjax` package consists of several modules, many of which rely on functionality from the `genjax.core` module, and build upon datatypes, transforms, and generative datatypes which are documented there. Generative function languages use the core datatypes and transforms to implement the generative function interface. Inference and learning algorithms are then implemented using the interface.

* [The core documentation](core/index.md) discusses key datatypes and transformations, which are used throughout the codebase.
* [The documentation on generative function languages](generative_functions/index.md) describes the functionality and usage for several generative function implementations, including distributions, a function-like language with primitives that allow callee generative functions, and combinator languages which provide structured patterns of control flow.
<!-- * [The inference documentation](inference/index.md) provides information on the standard inference library algorithms.
* [The differentiable programming documentation](diff_prog/index.md) describes GenJAX's approach to stateful computation and learning. -->
* [The documentation on extension modules](extensions/index.md) describes how users can extend GenJAX with new generative functions and inference functionality, while depending on 3rd party libraries.

# Library reference

This is the API documentation for modules and symbols which are exposed publicly from `genjax`.

At a high level, the `genjax` package consists of several modules, many of which rely on functionality from the `genjax.core` module, and build upon datatypes, transformation infrastructure, and generative datatypes which are defined there. Generative function languages use the core datatypes and infrastructure to implement the generative function interface. Inference and learning algorithms are then implemented using the interface.

Below we list several useful places to start, depending on your questions:

* [The core documentation](core/index.md) discusses key datatypes and transformations, which are used throughout the codebase. The generative function interface is documented here.
* [The documentation on generative function languages](generative_functions/index.md) describes the functionality and usage for several generative function implementations, including distributions, a the programmatic `static` language, and the generative function combinators.
* [The inference documentation](inference/index.md) provides information on the standard inference library algorithms, which, for now - focuses on sequential Monte Carlo (SMC).

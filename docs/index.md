#

<p align="center">
<img width="500px" src="./assets/img/logo.png"/>
</p>
<p align="center">
  <strong>
    Probabilistic programming with (parallel & differentiable) programmable inference.
  </strong>
</p>

---

You've arrived at the GenJAX documentation! This is the API documentation for modules and symbols which are exposed publicly from `genjax`.

The `genjax` package consists of several modules, many of which rely on functionality from the `genjax.core` module, and build upon the datatypes, transformation interpreters, and generative datatypes which are defined there. [Generative function languages](./generative_functions/) use the core datatypes and infrastructure to implement the generative function interface. [Inference algorithms](./inference/) are then implemented using the interface (and inference algorithms which utilize _learning_ (like variational inference) also use a [new extension to forward mode AD](./adev.md)).

Here are some useful places to start, depending on your questions:

* [The core documentation](core/index.md) discusses key datatypes and transformations, which are used throughout the codebase. The generative function interface is documented here, along with how GenJAX utilizes JAX (especially Pytrees), and the _masking_ system.
* [The documentation on generative function languages](generative_functions/index.md) describes the functionality and usage for several generative function implementations which ship with GenJAX, including distributions, the programmatic `static` language, generative function combinators, and the pedagogical `interpreted` language.
* [The inference documentation](inference/index.md) provides information on the key inference concepts, as well as the inference modules, including sequential Monte Carlo, and variational inference.

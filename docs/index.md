# Overview

**Short: it's a probabilistic programming library. It's primary computational object (the generative function) supports a well-studied and useful formal interface. It empowers users with the ability to customize and optimize their model and inference algorithms.**

---

**Gen** is a multi-paradigm (generative, differentiable, incremental) system for probabilistic programming. **GenJAX** is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate **generative functions** (1) (computational objects which represent probability measures over structured sample spaces) on native devices, accelerators, and other parallel fabrics. 
{ .annotate }

1.  By design, generative functions expose a concise interface for expressing approximate and differentiable inference algorithms. 

    The set of generative functions is extensible! You can implement your own - allowing advanced users to performance optimize their critical modeling/inference code paths.

    You can (and we, at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu/), do!) use these objects for machine learning - including robotics, natural language processing, reasoning about agents, and modelling / creating systems which exhibit human-like reasoning.

    A precise mathematical formulation of generative functions is given in [Marco Cusumano-Towner's PhD thesis][marco_thesis].

## Why Gen?

GenJAX is a [Gen][gen] implementation. If you're considering using GenJAX, or why this library exists - it's worth starting by understanding why Gen exists. Gen exists because probabilistic modeling and inference is hard - both computationally, and existing tooling.

[license]: license
[contributor guide]: contributing
[command-line reference]: usage
[gen]: https://www.gen.dev/
[gen.jl]: https://github.com/probcomp/Gen.jl
[genjax]: https://github.com/probcomp/genjax
[jax]: https://github.com/google/jax
[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf

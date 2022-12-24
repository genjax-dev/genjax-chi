# Overview

  [GenJAX][genjax] is system for probabilistic programming constructed by combining the concepts of [Gen][gen] with the hardware accelerator compilation capabilities of [JAX][jax].

Gen is a multi-paradigm (generative, incremental, differentiable) system for probabilistic programming. GenJAX is an implementation of Gen on top of JAX - it exposes the ability to programmatically construct and manipulate _generative functions_
(c.f. [](genjax/gen_fn): computational objects which represent probability measures over structured sample spaces. By construction, these objects expose a concise interface for expressing approximate and differentiable inference algorithms. The interface supports extension - allowing gradual performance optimization of critical modeling/inference code paths. 

You can (and we, at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu/), do!) use these objects for machine learning - including robotics, natural language processing, reasoning about agents, and modelling / creating systems which exhibit human-like reasoning.

A precise mathematical formulation of generative functions is given in [Marco Cusumano-Towner's PhD thesis][marco_thesis].

```{admonition} Novice
If you don't mind perusing carefully crafted documentation
(albeit in another language), you might also enjoy the [Gen.jl][gen.jl]
Julia documentation.
```

```{toctree}
---
hidden:
maxdepth: 1
caption: Getting started
---

genjax/gen_fn
genjax/diff_jl
genjax/notebooks
```

```{toctree}
---
hidden:
maxdepth: 1
caption: Library module reference
---

genjax/library/generative_functions/generative_functions
```

```{toctree}
---
hidden:
maxdepth: 1
caption: For developers
---

contributing
Code of Conduct <codeofconduct>
License <license>
Changelog <https://github.com/probcomp/genjax/releases>
```

[license]: license
[contributor guide]: contributing
[command-line reference]: usage
[gen]: https://www.gen.dev/
[gen.jl]: https://github.com/probcomp/Gen.jl
[genjax]: https://github.com/probcomp/genjax
[jax]: https://github.com/google/jax
[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf

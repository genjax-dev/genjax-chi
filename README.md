<br>
<p align="center">
<img width="400px" src="logo.png"/>
</p>
<br>

<div align="center">
<b><i>Probabilistic programming with Gen, built on top of JAX.</i></b>
</div>
<br>

[![Build Status](https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg)](https://github.com/probcomp/genjax/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://probcomp.github.io/genjax/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- Generative functions (models) are represented as pure functions from `(PRNGKey, *args)` to `(PRNGKey, retval)`.
- Exposes [the generative function interface](https://www.gen.dev/stable/ref/gfi/) as staged effect handlers built on top of `jax`.

  | Interface     | Semantics (informal)                                                                |
  | ------------- | ----------------------------------------------------------------------------------- |
  | `simulate`    | Sample from normalized measure over choice maps                                     |
  | `importance`  | Importance sample from conditioned measure, and compute an importance weight        |
  | `update`      | Given a new set of arguments and choice map, compute an updated trace               |
  | `arg_grad`    | Compute gradient of `logpdf` of choice map with respect to arguments                |
  | `choice_grad` | Compute gradient of `logpdf` of choice map with respect to values inside choice map |

- Supports usage of any computations acceptable by JAX (tbd) within generative function programs.

<div align="center">
<b>(Early stage)</b> 🔪 expect sharp edges 🔪
</div>

## Building + tour

This project uses [poetry](https://python-poetry.org/) for dependency management, and [nox](https://nox.thea.codes/en/stable/) to automate testing/linting/building.

Make sure these are installed and on path with a Python environment `^3.10.0`. Running `nox` will evaluate the full test/linting/build suite.

```
# Install dependencies, and run the tour example!
poetry install
poetry run python examples/tour.py
```

<div align="center">
<a href="/examples/tour.py">Jump into the tour!</a>
</div>

## References

Many bits of knowledge have gone into this project -- [you can find many of these bits at the MIT Probabilistic Computing Project page](http://probcomp.csail.mit.edu/) under publications. Here's an abbreviated list of high value references:

* [Marco Cusumano-Towner's thesis on Gen](https://www.mct.dev/assets/mct-thesis.pdf)
* [The main Gen.jl repository](https://github.com/probcomp/Gen.jl)
* (Trace types) [Alex Lew's paper on trace types](https://dl.acm.org/doi/10.1145/3371087)

---

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

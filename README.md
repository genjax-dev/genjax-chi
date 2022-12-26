<br>
<p align="center">
<img width="400px" src="docs/_static/assets/logo.png"/>
</p>
<br>

<div align="center">
<b><i>Probabilistic programming with Gen, built on top of JAX.</i></b>
</div>
<br>

[![Build Status](https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg)](https://github.com/probcomp/genjax/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://probcomp.github.io/genjax/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Public API: beartyped](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)!

<div align="center">
<b>(Early stage)</b> 🔪 expect sharp edges 🔪
</div>

## 🔎 What is it?

Gen is a multi-paradigm (generative, differentiable, incremental) language for probabilistic programming focused on [**generative functions**: computational objects which represent probability measures over structured sample spaces](https://probcomp.github.io/genjax/notebooks/introduction/intro_to_genjax/intro_to_genjax.html#what-is-a-generative-function). 

GenJAX is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate generative functions, as well as [JIT compile + auto-batch inference computations using generative functions onto GPU devices](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

<div align="center">
<a href="https://probcomp.github.io/genjax/notebooks/index.html">Jump into the notebooks!</a>
<br>
<br>
</div>

> GenJAX is part of a larger ecosystem of probabilistic programming tools based upon Gen. [Explore more...](https://www.gen.dev/)

## Development environment

This project uses [poetry](https://python-poetry.org/) for dependency management, [nox](https://nox.thea.codes/en/stable/) to automate testing/linting/building, and [quarto](https://quarto.org/) to render Jupyter notebooks for tutorial documentation.

Make sure these are installed and on path with a Python environment `^3.10.0`. Running `nox` will evaluate the full test/linting/build/docs suite.

### Environment setup script

Here's a simple script to setup a compatible development environment - if you can run this script, you have a working development environment which can be used to execute the notebooks, etc.

```bash
conda create --name genjax-py311 python=3.11 --channel=conda-forge
conda activate genjax-py311
pip install poetry
pip install nox
pip install nox-poetry
git clone https://github.com/probcomp/genjax
cd genjax
poetry lock
poetry install
poetry run jupyter-lab
```

## References

Many bits of knowledge have gone into this project -- [you can find many of these bits at the MIT Probabilistic Computing Project page](http://probcomp.csail.mit.edu/) under publications. Here's an abbreviated list of high value references:

- [Marco Cusumano-Towner's thesis on Gen](https://www.mct.dev/assets/mct-thesis.pdf)
- [The main Gen.jl repository](https://github.com/probcomp/Gen.jl)
- (Trace types) [Alex Lew's paper on trace types](https://dl.acm.org/doi/10.1145/3371087)
- (Prox) [Alex Lew's paper on recursive auxiliary-variable inference (RAVI)](https://arxiv.org/abs/2203.02836)
- (GenProx) [Alex Lew's Gen.jl implementation of Prox](https://github.com/probcomp/GenProx.jl)

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

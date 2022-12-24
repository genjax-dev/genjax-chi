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

<div align="center">
<b>(Early stage)</b> 🔪 expect sharp edges 🔪
</div>

## Building + notebooks

This project uses [poetry](https://python-poetry.org/) for dependency management, [nox](https://nox.thea.codes/en/stable/) to automate testing/linting/building, and [quarto](https://quarto.org/) to render Jupyter notebooks for tutorial documentation.

Make sure these are installed and on path with a Python environment `^3.10.0`. Running `nox` will evaluate the full test/linting/build/docs suite.

<div align="center">
<a href="https://probcomp.github.io/genjax/notebooks/index.html">Jump into the notebooks!</a>
<br>
<br>
</div>

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

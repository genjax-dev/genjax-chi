<br>
<p align="center">
<img width="400px" src="./docs/assets/img/logo.png"/>
</p>
<br>

<div align="center">
<b><i>Probabilistic programming with Gen, built on top of JAX.</i></b>
</div>
<br>

<div align="center">

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://probcomp.github.io/genjax/)
[![][jax_badge]](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Public API: beartyped](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

|          **Build Status**          |
| :--------------------------------: |
| [![][build_action_badge]][actions] |

</div>

[build_action_badge]: https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg
[actions]: https://github.com/probcomp/genjax/actions

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

### JAX influences

This project has several JAX-based influences. Here's an abbreviated list:

- [This notebook on static dispatch (author unknown)](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3)
- [Equinox (Patrick Kidger's work on neural networks via callable Pytrees)](https://github.com/patrick-kidger/equinox)
- [Oryx (interpreters and interpreter design)](https://github.com/jax-ml/oryx)

### Acknowledgements

The maintainers of this library would like to acknowledge the JAX team as well as the Oryx team for useful discussions and reference code for interpreter-based transformation patterns.

---

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

[jax_badge]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC

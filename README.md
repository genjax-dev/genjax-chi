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

[![PyPI](https://img.shields.io/pypi/v/genjax)](https://pypi.org/project/GenJAX/)
[![][jax_badge]](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Public API: beartyped](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg?style=flat-square)](https://beartype.readthedocs.io)

| **Documentation** |          **Build status**          |
| :---------------: | :--------------------------------: |
| [![](https://img.shields.io/badge/docs-stable-blue.svg?style=flat-square)](https://probcomp.github.io/genjax/) [![](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat-square&logo=jupyter&logoColor=white)](https://probcomp.github.io/genjax/notebooks/) | [![][main_build_action_badge]][main_build_status_url] [![][nightly_build_action_badge]][nightly_build_status_url] |

</div>

[main_build_action_badge]: https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg?style=flat-square&branch=main
[nightly_build_action_badge]: https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg?style=flat-square&branch=nightly
[actions]: https://github.com/probcomp/genjax/actions
[main_build_status_url]: https://github.com/probcomp/genjax/actions/workflows/ci.yml?query=branch%3Amain
[nightly_build_status_url]: https://github.com/probcomp/genjax/actions/workflows/ci.yml?query=branch%3Anightly


<div align="center">
<b>(Early stage)</b> 🔪 expect sharp edges 🔪
</div>

## 🔎 What is GenJAX?

Gen is a multi-paradigm (generative, differentiable, incremental) language for probabilistic programming focused on [**generative functions**: computational objects which represent probability measures over structured sample spaces](https://probcomp.github.io/genjax/notebooks/concepts/introduction/intro_to_genjax.html#what-is-a-generative-function).

GenJAX is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate generative functions, as well as [JIT compile + auto-batch inference computations using generative functions onto GPU devices](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

<div align="center">
<a href="https://probcomp.github.io/genjax/notebooks/index.html">Jump into the notebooks!</a>
<br>
<br>
</div>

> GenJAX is part of a larger ecosystem of probabilistic programming tools based upon Gen. [Explore more...](https://www.gen.dev/)

## Quickstart

Install GenJAX via [PyPI](https://pypi.org/project/genjax/):

```sh
pip install genjax
```

Then install [JAX](https://github.com/google/jax) using [this
guide](https://jax.readthedocs.io/en/latest/installation.html) to choose the
command for the architecture you're targeting. To run GenJAX without GPU
support:

```sh
pip install jax[cpu]==0.4.20
```

On a Linux machine with a GPU, run either of the following commands, depending
on which CUDA version (11 or 12) you have installed:

```sh
pip install jax[cuda11_pip]==0.4.20
pip install jax[cuda12_pip]==0.4.20
```

### Quick example

The following code snippet defines a generative function called `beta_bernoulli` that

- takes a shape parameter `beta`
- uses this to create and draw a value `p` from a [Beta
  distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- Flips a coin that returns 1 with probability `p`, 0 with probability `1-p` and
  returns that value

JIT-compiles a generative function interface method with JAX and then runs it with GenJAX:

```python
import genjax
import jax

@genjax.static_gen_fn
def beta_bernoulli(beta):
    p = genjax.beta(0.0, beta) @ "p"
    v = genjax.bernoulli(p) @ "v"
    return v

key = jax.random.PRNGKey(314159)
trace = jax.jit(beta_bernoulli.simulate)(key, (0.5, ))
choices = trace.get_choices()
```

`choices` is a record of all random choices made during the execution of the
generative function `beta_bernoulli`. Print it with a `genjax.console()`
instance:

```python
console = genjax.console()
console.print(choices)
```

resulting in:

```
(HierarchicalChoiceMap)
├── :p
│   └── (ValueChoice)
│       └──  f32[]
└── :v
    └── (ValueChoice)
        └──  i32[]
```

## References

Many bits of knowledge have gone into this project -- [you can find many of these bits at the MIT Probabilistic Computing Project page](http://probcomp.csail.mit.edu/) under publications. Here's an abbreviated list of high value references:

- [Marco Cusumano-Towner's thesis on Gen][marco_thesis]
- [The main Gen.jl repository][gen_jl]
- (Trace types) [(Lew et al) trace types][trace_types]
- (RAVI) [(Lew et al) Recursive auxiliary-variable inference][ravi]
- (GenSP) [Alex Lew's Gen.jl implementation of GenSP][gen_sp]
- (ADEV) [(Lew & Huot, et al) Automatic differentiation of expected values of probabilistic programs][adev]

### JAX influences

This project has several JAX-based influences. Here's an abbreviated list:

- [This notebook on static dispatch (Dan Piponi)][effect_handling_interp]
- [Equinox (Patrick Kidger's work on neural networks via callable Pytrees)][equinox]
- [Oryx (interpreters and interpreter design)][oryx]

### Acknowledgements

The maintainers of this library would like to acknowledge the JAX and Oryx maintainers for useful discussions and reference code for interpreter-based transformation patterns.

---

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf
[gen_jl]: https://github.com/probcomp/Gen.jl
[trace_types]: https://dl.acm.org/doi/10.1145/3371087
[adev]: https://arxiv.org/abs/2212.06386
[ravi]: https://arxiv.org/abs/2203.02836
[gen_sp]: https://github.com/probcomp/GenSP.jl
[effect_handling_interp]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[equinox]: https://github.com/patrick-kidger/equinox
[oryx]: https://github.com/jax-ml/oryx
[jax_badge]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC

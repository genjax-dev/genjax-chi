#

<p align="center">
<img width="500px" src="./assets/img/logo.png"/>
</p>
<p align="center">
  <strong>
    Probabilistic programming with (parallel & differentiable) programmable inference.
  </strong>
</p>

## 🔎 What is GenJAX?

Gen is a multi-paradigm (generative, differentiable, incremental) language for probabilistic programming focused on [**generative functions**: computational objects which represent probability measures over structured sample spaces](https://genjax.gen.dev/cookbook/active/intro.html#generative-functions).

GenJAX is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate generative functions, as well as [JIT compile + auto-batch inference computations using generative functions onto GPU devices](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

<div align="center">
<a href="https://genjax.gen.dev/cookbook/">Jump into the notebooks!</a>
<br>
<br>
</div>

> GenJAX is part of a larger ecosystem of probabilistic programming tools based upon Gen. [Explore more...](https://www.gen.dev/)

## Quickstart


GenJAX is currently private. To configure your machine to access the package,

- Run `/invite-genjax <google-account-email>` in any channel in the the probcomp Slack, or [file a ticket requesting access to the GenJAX-Users
group](https://github.com/probcomp/genjax/issues/new?assignees=sritchie&projects=&template=access.md&title=%5BACCESS%5D)
- [install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install)
- follow the instructions on the [installation page](https://cloud.google.com/sdk/docs/install)
- run `gcloud auth application-default login` as described [in this guide](https://cloud.google.com/sdk/docs/initializing).

To install GenJAX using `pip`:

```bash
pip install keyring keyrings.google-artifactregistry-auth
pip install genjax --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

To install a specific version:

```bash
pip install "genjax==<the_version>" --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

If you're using Poetry:

```bash
poetry self update && poetry self add keyrings.google-artifactregistry-auth
poetry source add --priority=explicit gcp https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
poetry add genjax --source gcp
```
Then install [JAX](https://github.com/google/jax) using [this
guide](https://jax.readthedocs.io/en/latest/installation.html) to choose the
command for the architecture you're targeting. To run GenJAX without GPU
support:

```sh
pip install jax[cpu]~=0.4.24
```

On a Linux machine with a GPU, run the following command:

```sh
pip install jax[cuda12]~=0.4.24
```

### Quick example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://console.cloud.google.com/vertex-ai/colab/notebooks?project=probcomp-caliban&activeNb=projects%2Fprobcomp-caliban%2Flocations%2Fus-west1%2Frepositories%2F09be0f8e-ccfd-4d34-a029-fed94d455c48)


The following code snippet defines a generative function called `beta_bernoulli` that

- takes a shape parameter `beta`
- uses this to create and draw a value `p` from a [Beta
  distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- Flips a coin that returns 1 with probability `p`, 0 with probability `1-p` and
  returns that value

Then, we create an inference problem (by specifying a posterior target), and utilize sampling
importance resampling to give produce single sample estimator of `p`.

We can JIT compile that entire process, run it in parallel, etc - which we utilize to produce an estimate for `p`
over 50 independent trials of SIR (with K = 50 particles).

```python
import jax
import jax.numpy as jnp
import genjax
from genjax import beta, flip, gen, Target, ChoiceMap
from genjax.inference.smc import ImportanceK

# Create a generative model.
@gen
def beta_bernoulli(α, β):
    p = beta(α, β) @ "p"
    v = flip(p) @ "v"
    return v

@jax.jit
def run_inference(obs: bool):
    # Create an inference query - a posterior target - by specifying
    # the model, arguments to the model, and constraints.
    posterior_target = Target(beta_bernoulli, # the model
                              (2.0, 2.0), # arguments to the model
                              ChoiceMap.d({"v": obs}), # constraints
                            )

    # Use a library algorithm, or design your own - more on that in the docs!
    alg = ImportanceK(posterior_target, k_particles=50)

    # Everything is JAX compatible by default.
    # JIT, vmap, to your heart's content.
    key = jax.random.key(314159)
    sub_keys = jax.random.split(key, 50)
    _, p_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, posterior_target
    )

    # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
    return jnp.mean(p_chm["p"])

(run_inference(True), run_inference(False))
```

```python
(Array(0.6039314, dtype=float32), Array(0.3679334, dtype=float32))
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

[actions]: https://github.com/probcomp/genjax/actions
[adev]: https://arxiv.org/abs/2212.06386
[cookbook]: https://genjax.gen.dev/cookbook/
[coverage_badge]: https://github.com/probcomp/genjax/coverage.svg
[effect_handling_interp]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[equinox]: https://github.com/patrick-kidger/equinox
[gen_jl]: https://github.com/probcomp/Gen.jl
[gen_sp]: https://github.com/probcomp/GenSP.jl
[jax_badge]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC
[main_build_action_badge]: https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg?style=flat-square&branch=main
[main_build_status_url]: https://github.com/probcomp/genjax/actions/workflows/ci.yml?query=branch%3Amain
[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf
[oryx]: https://github.com/jax-ml/oryx
[ravi]: https://arxiv.org/abs/2203.02836
[trace_types]: https://dl.acm.org/doi/10.1145/3371087

# Inference

## Sequential Monte Carlo

**Sequential Monte Carlo** (SMC) ([Del Moral, 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641)) is a class of Monte Carlo algorithms that is used to sample sequentially from a sequence of unnormalized target distributions, as well as estimate other quantities which are difficult to compute analytically, such as the normalizing constants of the targets of the sequence.

### Code example

We'll start with a simple example from this algorithm family: importance sampling.

#### Importance sampling

```python exec="yes" source="tabbed-left" session="ex-smc"
import jax
import genjax
from genjax import choice_map, normal, static_gen_fn
from genjax.inference import  Target, marginal
from genjax.inference.smc import ImportanceK
from genjax.typing import typecheck

console = genjax.console()

# Define a model.
@static_gen_fn
def model():
  x = normal(0.0, 1.0) @ "x"
  y = normal(x, 1.0) @ "y"

# Define a proposal, a `Marginal` targeting the latent variable `x`.
@marginal
@static_gen_fn
def proposal(target: Target):
  y = target["y"]
  _ = normal(y, 1.0) @ "x"

# Define an SMC algorithm, `ImportanceK`, with 5 particles.
target = Target(model, (), choice_map({"y" : 3.0}))
algorithm = ImportanceK(target, proposal, 5)

# Run importance sampling using the SMC interface on `ImportanceK`.
key = jax.random.PRNGKey(314159)
particle_collection = jax.jit(algorithm.run_smc)(key)

# Print the log marginal likelihood estimate.
print(console.render(particle_collection.get_log_marginal_likelihood_estimate()))
```

Importance sampling is often used to initialize a particle collection, which can be evolved further through subsequent SMC steps (like extension moves and resampling).

#### Sampling importance resampling (SIR)

By virtue of the `SMCAlgorithm` interface, any SMC algorithm also implement single particle resampling when utilizing the `Distribution` interfaces:

```python exec="yes" source="tabbed-left" session="ex-smc"
Z, choice = jax.jit(algorithm.random_weighted)(key, target)
print(console.render(choice))
```

So `ImportanceK.random_weighted` exposes the SIR algorithm, and returns an estimate of the marginal likelihood (`Z`, above) and a sample from the final weighted particle_collection (`choice`, above).

### Module reference

::: genjax._src.inference.smc
    options:
      members:
        - ParticleCollection
        - SMCAlgorithm
        - Importance
        - ImportanceK
        - ChangeTarget
      show_root_heading: true


## Variational inference


<figure markdown="span">
  ![GenJAX VI architecture](../../assets/img/genjax-vi.png){ width = "300" }
  <figcaption><b>Fig. 1</b>: How variational inference works in GenJAX.</figcaption>
</figure>

**Variational inference** ([Blei et al, 2016](https://arxiv.org/abs/1601.00670)) is an approximate inference technique where the problem of computing the posterior distribution $P'$ is transformed into an optimization problem. The idea is to find a distribution $Q$ that is close to the true posterior $P'$ by minimizing the Kullback-Leibler (KL) divergence between the two distributions.

GenJAX provides automation for this process by exposing unbiased gradient automation based on the stack shown in **Fig. 1**. At a high level, the stack illustrates that `genjax.inference.vi` utilizes implementations of generative interfaces like $\textbf{sim}\{ \cdot \}$ and $\textbf{density}\{ \cdot \}$ [in the source language of a differentiable probabilistic language called ADEV](../adev.md).

ADEV is a new extension to automatic differentiation which adds supports for _expectations_ - so when we provide implementations using ADEV's language, we gain the ability to automatically derive unbiased gradient estimators for expected value objectives.

### Code example

Here's a small example using the library loss `genjax.vi.ELBO`:

```python exec="yes" source="tabbed-left" session="ex-vi"
import jax
import genjax
from genjax import choice_map, normal, static_gen_fn
from genjax.inference import  Target, marginal
from genjax.inference.vi import ELBO, normal_reparam
from genjax.typing import typecheck

console = genjax.console()

@static_gen_fn
def model(v):
  x = normal(0.0, 1.0) @ "x"
  y = normal(x, 1.0) @ "y"

# The guide uses special (ADEV) differentiable generative function primitives.
@marginal
@static_gen_fn
def guide(target: Target):
  (v, ) = target.args
  x = normal_reparam(v, 1.0) @ "x"

# Using a library loss.
elbo = ELBO(
  guide,
  lambda v: Target(model, (v, ), choice_map({"y": 3.0})),
)

# Output has the same Pytree shape as input arguments to `ELBO.grad_estimate`,
# excluding the key.
key = jax.random.PRNGKey(314159)
(v_grad,) = jax.jit(elbo.grad_estimate)(key, (1.0, ))
print(console.render(v_grad))
```

Let's examine the construction of the `ELBO` instance:

```python exec="yes" source="tabbed-left" session="ex-vi"
elbo = ELBO(
  # Approximation to the target.
  guide,
  # The posterior target -- can also have learnable parameters!
  lambda v: Target(model, (v, ), choice_map({"y": 3.0})),
)
```
The signature of `ELBO` allows the user to specify what "to focus on" in the `ELBO.grad_estimate` interface. For example, let's say we also have a learnable model, which accepts a parameter `p` which we'd like to learn -- we can modify the `Target` lambda:

```python exec="yes" source="tabbed-left" session="ex-vi"
@marginal
@static_gen_fn
def guide(target: Target):
  (_, v) = target.args
  x = normal_reparam(v, 1.0) @ "x"

@static_gen_fn
def model(p, v):
  x = normal(p, 1.0) @ "x"
  y = normal(x, 1.0) @ "y"

elbo = ELBO(
  # Approximation to the target.
  guide,
  # The posterior target -- p is learnable!
  lambda p, v: Target(model, (p, v), choice_map({"y": 3.0})),
)
(p_grad, v_grad) = jax.jit(elbo.grad_estimate)(key, (1.0, 1.0))
print(console.render((p_grad, v_grad)))
```

### Module reference

::: genjax._src.inference.vi
    options:
      members:
        - ADEVDistribution
        - ExpectedValueLoss
        - ELBO
      show_root_heading: true

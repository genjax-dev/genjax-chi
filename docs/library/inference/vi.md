# Variational inference


<figure markdown="span">
  ![GenJAX VI architecture](../../assets/img/genjax-vi.png){ width = "300" }
  <figcaption><b>Fig. 1</b>: How variational inference works in GenJAX.</figcaption>
</figure>

**Variational inference** ([Blei et al, 2016](https://arxiv.org/abs/1601.00670)) is an approximate inference technique where the problem of computing the posterior distribution $P'$ is transformed into an optimization problem. The idea is to find a distribution $Q$ that is close to the true posterior $P'$ by minimizing the Kullback-Leibler (KL) divergence between the two distributions.

GenJAX provides automation for this process by exposing unbiased gradient automation based on the stack shown in **Fig. 1**. At a high level, the stack illustrates that `genjax.inference.vi` utilizes implementations of generative interfaces like $\textbf{sim}\{ \cdot \}$ and $\textbf{density}\{ \cdot \}$ [in the source language of a differentiable probabilistic language called ADEV](../adev.md).

ADEV is a new extension to automatic differentiation which adds supports for _expectations_ - so when we provide implementations using ADEV's language, we gain the ability to automatically derive unbiased gradient estimators for expected value objectives.

## Code example

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

## Module reference

::: genjax._src.inference.vi
    options:
      members:
        - ADEVDistribution
        - ExpectedValueLoss
        - ELBO
      show_root_heading: true

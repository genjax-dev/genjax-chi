# Automatic differentiation of expected values

![Overview of ADEV](../assets/img/adev-diagram.png)

## Overview of ADEV

ADEV ([Lew and Huot, et al, 2023](https://dl.acm.org/doi/abs/10.1145/3571198)) is a method of automatically differentiating loss functions defined as _expected values_ of probabilistic processes. ADEV users define a _probabilistic program_ $t$, which, given a parameter of type $\mathbb{R}$ (or a subtype), outputs a value of type $\widetilde{\mathbb{R}}$,
which represents probabilistic estimators of losses. We translate $t$ to a new probabilistic program $s$,
whose expected return value is the derivative of $t$’s expected return value. Running $s$ yields provably unbiased
estimates $x_i$ of the loss's derivative, which can be used in the inner loop of stochastic optimization algorithms like ADAM or stochastic gradient descent.

ADEV goes beyond standard AD by explicitly supporting probabilistic primitives (like `flip`, for flipping a coin). If these probabilistic constructs are ignored, standard AD may produce incorrect results, as this figure from our paper illustrates:

![Optimizing an example loss function using ADEV](../assets/img/example.png)

In this example, standard AD fails to account for the parameter $\theta$'s effect on the _probability_ of entering each branch. ADEV, by contrast, correctly accounts
for the probabilistic effects, generating similar code to what a practitioner might hand-derive. Correct
gradients are often crucial for downstream applications, e.g. optimization via stochastic gradient descent.

ADEV compositionally supports various gradient estimation strategies from the literature, including:

- Reparameterization trick (Kingma & Welling 2014)
- Score function estimator (Ranganath et al. 2014)
- Baselines as control variates (Mnih and Gregor 2014)
- Multi-sample estimators that Storchastic supports (e.g. leave-one-out baselines) (van Krieken et al. 2021)
- Variance reduction via dependency tracking (Schulman et al. 2015)
- Special estimators for differentiable particle filtering (Ścibior et al. 2021)
- Implicit reparameterization (Figurnov et al. 2018)
- Measure-valued derivatives (Heidergott and Vázquez-Abad 2000)
- Reparameterized rejection sampling (Nasseth et al. 2017)

## Our implementation

**Other implementations: [Haskell](https://github.com/probcomp/adev), [Julia](https://github.com/probcomp/ADEV.jl)**

GenJAX provides an implementation of ADEV via the `genjax.adev` module, and uses ADEV's AD algorithm to automate the construction of unbiased gradient estimators for the loss functions exposed by `genjax.vi` (including ELBO, IWELBO, ingredients of wake-sleep, etc).

### Current limitations

* Generative code (meaning sampling from ADEV primitives) is currently not correctly accounted for inside of JAX's higher order primitives (like `jax.lax.scan`, `jax.lax.cond`, etc).
* Batching of generative code (meaning sampling from ADEV primitives _within `jax.vmap`'d code_) is not correctly accounted for.

## Code example

This example illustrates self-contained usage of ADEV via `genjax.adev`. We expect most users won't interact with the soruce language directly, but will instead use ADEV via `genjax.vi`, and other learning abstractions built on top of it.

```python exec="yes" source="tabbed-left" session="ex-adev"
import jax
import jax.numpy as jnp

import genjax
from genjax.adev import expectation
from genjax.adev import flip_enum

# Sets up pretty printing + good stack traces.
console = genjax.console(max_frames=100, show_locals=False)

# The exact expectation is: (p² - p) / 2
# The exact gradient is: (p - 1/2)
@expectation
def flip_exact_loss(p):
    b = flip_enum(p)
    return jax.lax.cond(
        b,
        lambda _: 0.0,
        lambda p: -p / 2.0,
        p,
    )

key = jax.random.PRNGKey(314159)

# Forward mode.
v, p_tangent = jax.jit(flip_exact_loss.jvp_estimate)(
    key,  # PRNG key
    (0.7,),  # Primals
    (1.0,),  # Tangents
)
print(console.render((v, p_tangent)))

# Reverse mode (automatically derived by JAX).
p_grad = jax.jit(flip_exact_loss.grad_estimate)(
    key,  # PRNG key
    (0.7,),  # Primals
)
print(console.render(p_grad))
```

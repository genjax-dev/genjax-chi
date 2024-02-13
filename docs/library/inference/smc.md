# Sequential Monte Carlo

**Sequential Monte Carlo** (SMC) ([Del Moral, 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641)) is a class of Monte Carlo algorithms that is used to sample sequentially from a sequence of unnormalized target distributions, as well as estimate other quantities which are difficult to compute analytically, such as the normalizing constants of the targets of the sequence.

## Code example

We'll start with a simple example from this algorithm family: importance sampling.

### Importance sampling

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

### Sampling importance resampling (SIR)

By virtue of the `SMCAlgorithm` interface, any SMC algorithm also implement single particle resampling when utilizing the `Distribution` interfaces:

```python exec="yes" source="tabbed-left" session="ex-smc"
Z, choice = jax.jit(algorithm.random_weighted)(key, target)
print(console.render(choice))
```

So `ImportanceK.random_weighted` exposes the SIR algorithm, and returns an estimate of the marginal likelihood (`Z`, above) and a sample from the final weighted particle_collection (`choice`, above).

## Module reference

::: genjax._src.inference.smc
    options:
      members:
        - ParticleCollection
        - SMCAlgorithm
        - Importance
        - ImportanceK
        - ChangeTarget
      show_root_heading: true

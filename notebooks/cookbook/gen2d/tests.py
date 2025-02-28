import discrete_distributions
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import genjax
from genjax import ChoiceMapBuilder as C

key = jax.random.key(0)


### Test for discretized Gaussian

mean = 0
std_dev = 1
ncat = 64
discrete_values, probabilities = discrete_distributions.discretize_normal(
    mean, std_dev, ncat
)

# Test the discretization by sampling 100,000 from a normal distribution and from the discretized version
samples1 = tfp.distributions.Normal(mean, std_dev).sample(1000000, seed=key)
disc_samples1 = np.random.choice(discrete_values, size=100000, p=probabilities)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples1, bins=1000, density=True, alpha=0.7, label="Normal Distribution")
plt.hist(
    disc_samples1, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Normal and Discretized Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

### Test for discretized Gamma

shape = 2  # Shape parameter (k)
scale = 2  # Scale parameter (theta)
ncat = 64  # Number of discrete values
discrete_values, probabilities = discrete_distributions.discretize_gamma(
    shape, scale, ncat
)

# Test the discretization
samples2 = tfp.distributions.Gamma(shape, 1 / scale).sample(1000000, seed=key)
disc_samples2 = np.random.choice(discrete_values, size=100000, p=probabilities)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples2, bins=1000, density=True, alpha=0.7, label="Gamma Distribution")
plt.hist(
    disc_samples2, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Gamma and Discretized Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

### Test for discretized Inverse-Gamma

shape = 4  # Shape parameter (α)
scale = 4  # Scale parameter (β)
ncat = 64  # Number of discrete values
discrete_values, probabilities = discrete_distributions.discretize_inverse_gamma(
    shape, scale, ncat
)

# Test the discretization
samples3 = tfp.distributions.InverseGamma(shape, scale).sample(1000000, seed=key)
disc_samples3 = jax.random.choice(
    key, discrete_values, shape=(1000000,), p=probabilities
)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(
    samples3, bins=1000, density=True, alpha=0.7, label="Inverse Gamma Distribution"
)
plt.hist(
    disc_samples3, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Inverse-Gamma and Discretized Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

### Test for discretized gamma distribution in GenJAX


@genjax.gen
def model1():
    x = discrete_distributions.discrete_gamma(3.0, 3.0) @ "x"
    return x


tr = model1.simulate(key, ())
obs = C["x"].set(3.0)
w, _ = model1.assess(obs, ())
assert w == 0.0
obs2 = C["x"].set(tr.get_retval())
w, _ = model1.assess(obs2, ())
assert w != 0.0

### Test for discretized normal distribution in GenJAX


@genjax.gen
def model2():
    x = discrete_distributions.discrete_normal(3.0, 3.0) @ "x"
    return x


tr = model2.simulate(key, ())
obs = C["x"].set(3.0)
w, _ = model2.assess(obs, ())
assert w == 0.0
obs2 = C["x"].set(tr.get_retval())
w, _ = model2.assess(obs2, ())
assert w != 0.0

### Test for discretized inverse-gamma distribution in GenJAX


@genjax.gen
def model3():
    x = discrete_distributions.discrete_inverse_gamma(3.0, 3.0) @ "x"
    return x


tr = model3.simulate(key, ())
obs = C["x"].set(3.0)
w, _ = model3.assess(obs, ())
assert w == 0.0
obs2 = C["x"].set(tr.get_retval())
w, _ = model3.assess(obs2, ())
assert w != 0.0

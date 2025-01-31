import discrete_distributions
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp

key = jax.random.PRNGKey(0)

### Test for discretized Gaussian

mean = 0
std_dev = 1
ncat = 64
discrete_values, probabilities = discrete_distributions.discretize_normal(
    mean, std_dev, ncat
)

# Test the discretization by sampling 100,000 from a normal distribution and from the discretized version
samples = tfp.distributions.Normal(mean, std_dev).sample(1000000, seed=key)
disc_samples = np.random.choice(discrete_values, size=100000, p=probabilities)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=1000, density=True, alpha=0.7, label="Normal Distribution")
plt.hist(
    disc_samples, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
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
samples = tfp.distributions.Gamma(shape, scale).sample(1000000, seed=key)
samples = np.random.gamma(shape=shape, scale=scale, size=100000)
disc_samples = np.random.choice(discrete_values, size=100000, p=probabilities)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=1000, density=True, alpha=0.7, label="Normal Distribution")
plt.hist(
    disc_samples, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Normal and Discretized Distributions")
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
samples = tfp.distributions.InverseGamma(shape, scale).sample(1000000, seed=key)
disc_samples = jax.random.choice(
    key, discrete_values, shape=(1000000,), p=probabilities
)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=1000, density=True, alpha=0.7, label="Normal Distribution")
plt.hist(
    disc_samples, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Normal and Discretized Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

### Test for sampled-based discretized Inverse-Gamma

shape = 4  # Shape parameter (α)
scale = 4  # Scale parameter (β)
ncat = 64  # Number of discrete values
discrete_values, probabilities = (
    discrete_distributions.sampled_based_discretized_inverse_gamma(shape, scale, ncat)
)

# Test the discretization
samples = tfp.distributions.InverseGamma(shape, scale).sample(1000000, seed=key)
disc_samples = jax.random.choice(
    key, discrete_values, shape=(1000000,), p=probabilities
)

# Plot both on a graph
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=1000, density=True, alpha=0.7, label="Normal Distribution")
plt.hist(
    disc_samples, bins=ncat, density=True, alpha=0.7, label="Discretized Distribution"
)
plt.title("Comparison of Normal and Discretized Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

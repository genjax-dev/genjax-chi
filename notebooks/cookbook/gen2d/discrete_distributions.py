import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfps

import genjax
from genjax import Pytree
from genjax._src.core.typing import (
    PRNGKey,
)

tfd = tfps.distributions


def discretize_normal(mean, std_dev, ncat):
    # Create an array of discrete values centered around the mean
    lower_bound = mean - 4.0 * std_dev
    upper_bound = mean + 4.0 * std_dev
    step = (upper_bound - lower_bound) / (ncat - 1)

    def body(carry, x):
        return carry + step, carry + step

    _, discrete_values = jax.lax.scan(body, lower_bound, jnp.arange(ncat))

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = tfd.Normal(loc=mean, scale=std_dev).prob(discrete_values)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def discretize_gamma(shape, scale, ncat):
    # Calculate the mean and variance of the gamma distribution
    mean = shape * scale
    variance = shape * (scale**2)

    # Create an array of discrete values centered around the mean
    # Extend the range to capture the distribution
    lower_bound = jnp.maximum(0, mean - 4 * jnp.sqrt(variance))
    upper_bound = mean + 4 * jnp.sqrt(variance)
    step = (upper_bound - lower_bound) / (ncat - 1)

    def body(carry, x):
        return carry + step, carry + step

    _, discrete_values = jax.lax.scan(body, lower_bound, jnp.arange(ncat))

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = tfd.Gamma(concentration=shape, rate=1 / scale).prob(discrete_values)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def discretize_inverse_gamma(shape, scale, ncat):
    mean = scale / (shape - 1)
    variance = jax.lax.cond(
        shape > 2,
        lambda: (scale**2) / ((shape - 1.0) ** 2.0 * (shape - 2.0)),
        lambda: 0.0,
    )
    lower_bound = jax.lax.cond(
        shape > 2.0,
        lambda: jnp.maximum(0.01, mean - 4.0 * jnp.sqrt(variance)),
        lambda: 0.01,
    )
    upper_bound = jax.lax.cond(
        shape > 2.0, lambda: mean + 4.0 * jnp.sqrt(variance), lambda: mean + 10.0
    )
    step = (upper_bound - lower_bound) / (ncat - 1)

    def body(carry, x):
        return carry + step, carry + step

    _, discrete_values = jax.lax.scan(body, lower_bound, jnp.arange(ncat))

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = tfd.InverseGamma(concentration=shape, scale=scale).prob(
        discrete_values
    )

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def c(shape, scale, ncat, sample_size=100000):
    # Sample from the inverse gamma distribution
    key = jax.random.PRNGKey(0)
    samples = tfd.InverseGamma(concentration=shape, scale=scale).sample(
        sample_size, seed=key
    )

    # Create a histogram of the samples
    counts, bin_edges = jnp.histogram(samples, bins=ncat, density=True)

    # Calculate the midpoints of the bins for discrete values
    discrete_values = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Normalize the counts to get probabilities
    probabilities = counts / jnp.sum(counts)

    return discrete_values, probabilities


@genjax.Pytree.dataclass
class DiscretizedNormal(genjax.ExactDensity):
    nbins: int = Pytree.static(default=64)
    """
    Discretized Normal distribution.
    Args:
    - `mean` (float): the mean of the distribution.
    - `std_dev` (float): the standard deviation of the distribution.
    - `nbins` (int): the number of categories for discretization.

    Support:
    - The discrete values in the support.

    The probability of each discrete value is
    proportional to the PDF of the Normal distribution.
    """

    def sample(self, key: PRNGKey, *args):
        mean, std_dev = args
        discrete_values, probabilities = discretize_normal(mean, std_dev, self.nbins)
        idx = genjax.categorical(probabilities).simulate(key, ()).value
        return discrete_values[idx]

    def logpdf(self, x, *args):
        mean, std_dev = args
        discrete_values, probabilities = discretize_normal(mean, std_dev, self.nbins)
        return jnp.log(probabilities[discrete_values == x])


discrete_normal = DiscretizedNormal()


@genjax.Pytree.dataclass
class DiscretizedGamma(genjax.ExactDensity):
    nbins: int = Pytree.static(default=64)
    """
    Discretized Gamma distribution.
    Args:
    - `shape` (float): the shape parameter of the distribution.
    - `scale` (float): the scale parameter of the distribution.
    - `nbins` (int): the number of categories for discretization.

    Support:
    - The discrete values in the support.

    The probability of each discrete value is
    proportional to the PDF of the Gamma distribution.
    """

    def sample(self, key: PRNGKey, *args):
        shape, scale = args
        discrete_values, probabilities = discretize_gamma(shape, scale, self.nbins)
        idx = genjax.categorical(probabilities).simulate(key, ()).value
        return discrete_values[idx]

    def logpdf(self, x, *args):
        shape, scale = args
        discrete_values, probabilities = discretize_gamma(shape, scale, self.nbins)
        return jnp.log(probabilities[discrete_values == x])


discrete_gamma = DiscretizedGamma()


@genjax.Pytree.dataclass
class DiscretizedInverseGamma(genjax.ExactDensity):
    nbins: int = Pytree.static(default=64)
    """
    Discretized Inverse Gamma distribution.
    Args:
    - `shape` (float): the shape parameter of the distribution.
    - `scale` (float): the scale parameter of the distribution.
    - `nbins` (int): the number of categories for discretization.

    Support:
    - The discrete values in the support.

    The probability of each discrete value is
    proportional to the PDF of the Inverse Gamma distribution.
    """

    def sample(self, key: PRNGKey, *args):
        shape, scale = args
        discrete_values, probabilities = discretize_inverse_gamma(
            shape, scale, self.nbins
        )
        idx = genjax.categorical(probabilities).simulate(key, ()).value
        return discrete_values[idx]

    def logpdf(self, x, *args):
        shape, scale = args
        discrete_values, probabilities = discretize_inverse_gamma(
            shape, scale, self.nbins
        )
        return jnp.log(probabilities[discrete_values == x])


discrete_inverse_gamma = DiscretizedInverseGamma()

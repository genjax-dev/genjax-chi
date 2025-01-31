import jax.numpy as jnp
import scipy.stats as ss
from utils import FloatFromDiscreteSet, unwrap

import genjax
from genjax import Const


def sum_exponential_over_range(minx, maxx, log_base):
    """Sum of a geometric series: sum_{i=minx}^{maxx} exp(log_base)^i."""
    val = (jnp.exp(log_base * minx) - jnp.exp(log_base * (maxx + 1))) / (
        1 - jnp.exp(log_base)
    )
    return jnp.where(minx <= maxx, val, 0)


def normalizing_const_for_discretized_laplace_for_center_in_range(
    minx: int, maxx: int, center: int, scale: float
):
    """
    Returns sum_{x=minx}^{maxx} exp(-|x - center| / scale).
    Assumes minx <= center <= maxx.

    Possible improvements for future iterations:
    - Support the case where `center` is not an integer.
    """
    ## Derivation of the first computed expression:
    #   sum_{x=minx}^{center} exp(-|x - center| / scale)
    # = sum_{x=minx}^{center} exp(-(center - x) / scale)
    # = sum_{y = 0}^{center - minx} exp(-y / scale) [set y = center - x]
    p_minx_to_center = sum_exponential_over_range(0, center - minx, -1.0 / scale)

    ## Derivation of the second computed expression:
    #   sum_{x=center+1}^{maxx} exp(-|x - center| / scale)
    # = sum_{x=center+1}^{maxx} exp(-(x - center) / scale)
    # = sum_{y = 1}^{maxx - center} exp(-y / scale) [set y = x - center]
    p_center_plus_1_to_maxx = sum_exponential_over_range(1, maxx - center, -1.0 / scale)

    ## Final return is the sum of these two expressions.
    return p_minx_to_center + p_center_plus_1_to_maxx


def normalizing_const_for_discretized_laplace(
    minx: int, maxx: int, center: int, scale: float
):
    """
    Returns sum_{x=minx}^{maxx} exp(-|x - center| / scale).

    Possible improvements for future iterations:
    - Support the case where `center` is not an integer.
    """
    in_range = normalizing_const_for_discretized_laplace_for_center_in_range(
        minx, maxx, center, scale
    )

    ## Derivation of the first computed expression:
    #   sum_{x=minx}^{maxx} exp(-(x - center) / scale)
    # = sum_{y=minx - center}^{maxx - center} exp(-y / scale) [set y = x - center]
    p_if_center_below_minx = sum_exponential_over_range(
        minx - center, maxx - center, -1.0 / scale
    )

    ## Derivation of the second computed expression:
    #   sum_{x=minx}^{maxx} exp(-(center - x) / scale)
    # = sum_{y=center - maxx}^{center - minx} exp(-y / scale) [set y = center - x]
    p_if_center_above_maxx = sum_exponential_over_range(
        center - maxx, center - minx, -1.0 / scale
    )

    ## Final return is a 3-way among these two expressions,
    # and the expression for when center is in range.
    return jnp.where(
        minx <= center,
        jnp.where(center <= maxx, in_range, p_if_center_above_maxx),
        p_if_center_below_minx,
    )


def discretized_laplace_logpdf(x, center, scale, minx, maxx):
    assert isinstance(center, int) or jnp.issubdtype(center.dtype, jnp.integer), (
        "center must be an integer"
    )
    assert isinstance(x, int) or jnp.issubdtype(x.dtype, jnp.integer), (
        f"x must be an integer but x = {x}"
    )

    # convert to int32, in case these are given as uint8.
    # (if we keep these as uint8, arithmetic will be done mod 256,
    # which is not what we want)
    center = jnp.array(center, dtype=jnp.int32)
    x = jnp.array(x, dtype=jnp.int32)

    # PDF is exp(-|x - center| / scale) / Z
    Z = normalizing_const_for_discretized_laplace(minx, maxx, center, scale)
    return (-jnp.abs(x - center) / scale) - jnp.log(Z)


@genjax.Pytree.dataclass
class DiscretizedLaplace(genjax.ExactDensity):
    """
    Discretized Laplace distribution.
    Args:
    - `center` (int): the center of the distribution.
    - `scale` (float): the scale of the distribution.
    - `minx` (int or Const(int)): the minimum value in the support.
    - `maxx` (int or Const(int)): the maximum value in the support.

    Support:
    - The integers in [minx, maxx].

    The probability of each integer x is
    proportional to exp(-|x - center| / scale).

    This distribution is nice because PDF evaluations occur in O(1) time
    and do not require calling any special functions like erf or erfc.

    [I have not empirically tested the compute cost of erf/erfc, so it is possible
    the discretized normal version of this would also be just fine.]

    Possible improvements for future iterations:
    - Support the case where `center` is not an integer.
    - Investigate more efficient ways to sample from this distribution.
    """

    def sample(self, key, center, scale, minx: Const, maxx: Const):
        minx, maxx = unwrap(minx), unwrap(maxx)
        return (
            genjax.categorical(
                discretized_laplace_logpdf(
                    jnp.arange(minx, maxx + 1), center, scale, minx, maxx
                )
            )(key)
            + minx
        )

    def logpdf(self, x, center, scale, minx: Const, maxx: Const):
        minx, maxx = unwrap(minx), unwrap(maxx)
        return discretized_laplace_logpdf(x, center, scale, minx, maxx)

    @property
    def __doc__(self):
        return DiscretizedLaplace.__doc__


discretized_laplace = DiscretizedLaplace()


@genjax.Pytree.dataclass
class IndexSpaceDiscretizedLaplace(genjax.ExactDensity):
    """
    A distribution over `FloatFromDiscreteSet` objects,
    where the distribution over indices into the `FloatFromDiscreteSet`
    `Domain` is a discretized Laplace distribution.

    Distribution arguments:
    - `mean`: a `FloatFromDiscreteSet` object, giving the `Domain`
        for this distribution, and the mean value.
    - `scale`: the scale of the discretized Laplace distribution
        on indices into this domain.

    Distribution support:
    - The support is the set of all `FloatFromDiscreteSet` objects
        with the same `Domain` as `mean`.
    """

    def sample(self, key, mean: FloatFromDiscreteSet, scale):
        idx = discretized_laplace.sample(
            key, mean.idx, scale, 0, mean.domain.values.size - 1
        )
        return FloatFromDiscreteSet(idx=idx, domain=mean.domain)

    def logpdf(self, x, mean: FloatFromDiscreteSet, scale):
        return discretized_laplace.logpdf(
            x.idx, mean.idx, scale, 0, mean.domain.values.size - 1
        )

    @property
    def __doc__(self):
        return IndexSpaceDiscretizedLaplace.__doc__


index_space_discretized_laplace = IndexSpaceDiscretizedLaplace()


def discretize_normal(mean, std_dev, ncat):
    # Create an array of discrete values centered around the mean
    discrete_values = jnp.linspace(mean - 4 * std_dev, mean + 4 * std_dev, ncat)

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = ss.norm.pdf(discrete_values, loc=mean, scale=std_dev)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def discretize_gamma(shape, scale, ncat):
    # Calculate the mean and variance of the gamma distribution
    mean = shape * scale
    variance = shape * (scale**2)

    # Create an array of discrete values centered around the mean
    # Extend the range to capture the distribution
    discrete_values = jnp.linspace(
        max(0, mean - 4 * jnp.sqrt(variance)), mean + 4 * jnp.sqrt(variance), ncat
    )

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = ss.gamma.pdf(discrete_values, a=shape, scale=scale)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def discretize_gamma(shape, scale, ncat):
    # Calculate the mean and variance of the gamma distribution
    mean = shape * scale
    variance = shape * (scale**2)

    # Create an array of discrete values centered around the mean
    # Extend the range to capture the distribution
    discrete_values = jnp.linspace(
        max(0, mean - 4 * jnp.sqrt(variance)), mean + 4 * jnp.sqrt(variance), ncat
    )

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = ss.gamma.pdf(discrete_values, a=shape, scale=scale)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    return discrete_values, probabilities


def discretize_inverse_gamma(shape, scale, ncat):
    # Check if the shape parameter is valid
    if shape <= 1:
        raise ValueError(
            "Shape parameter must be greater than 1 for the mean to be defined."
        )

    elif shape > 2:
        # Calculate the mean and variance of the inverse gamma distribution
        mean = scale / (shape - 1)
        variance = (scale**2) / ((shape - 1) ** 2 * (shape - 2))

        # Create an array of discrete values centered around the mean
        # Ensure that we only generate positive values
        lower_bound = max(0, mean - 4 * jnp.sqrt(variance))
        upper_bound = mean + 4 * jnp.sqrt(variance)

        # Ensure lower_bound is positive
        if lower_bound < 0:
            lower_bound = 0.01  # Set a small positive value to avoid zero

    else:
        mean = scale / (shape - 1)
        lower_bound = 0.01  # Avoid zero
        upper_bound = mean + 10  # Extend the upper bound to capture the distribution

    discrete_values = jnp.linspace(lower_bound, upper_bound, ncat)

    # Calculate the probability density function (PDF) for each discrete value
    pdf_values = ss.invgamma.pdf(discrete_values, a=shape, scale=scale)

    # Normalize the PDF to sum to 1 (to create a probability distribution)
    probabilities = pdf_values / jnp.sum(pdf_values)

    # Check for NaN values in probabilities
    if jnp.any(jnp.isnan(probabilities)):
        raise ValueError(
            "NaN values found in probabilities. Check the input parameters."
        )

    return discrete_values, probabilities


def sampled_based_discretized_inverse_gamma(shape, scale, ncat, sample_size=100000):
    # Check if the shape parameter is valid
    if shape <= 1:
        raise ValueError(
            "Shape parameter must be greater than 1 for the mean to be defined."
        )

    # Sample from the inverse gamma distribution
    samples = ss.invgamma.rvs(a=shape, scale=scale, size=sample_size)

    # Create a histogram of the samples
    counts, bin_edges = jnp.histogram(samples, bins=ncat, density=True)

    # Calculate the midpoints of the bins for discrete values
    discrete_values = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Normalize the counts to get probabilities
    probabilities = counts / jnp.sum(counts)

    return discrete_values, probabilities


### TODO: turn these into GenJAX distributions using ExactDensity.

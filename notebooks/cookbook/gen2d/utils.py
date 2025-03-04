### Taken from gen3d/src/gen3d/variants/condor/utils.py

from dataclasses import dataclass, replace
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import genjax
from genjax import Const, PythonicPytree, Pytree

T = TypeVar("T")

_gamma = genjax.tfp_distribution(tfp.distributions.Gamma)


def sample_gamma_safe(key, alpha, beta):
    sample = _gamma.sample(key, alpha, beta)
    return jnp.where(sample == 0, 1e-12, sample)


gamma = genjax.exact_density(sample_gamma_safe, _gamma.logpdf)


def sample_dirichlet_safe(key, alpha):
    if alpha.shape == (1,):
        return jnp.array([1.0])
    sample = genjax.dirichlet.sample(key, alpha)
    return jnp.where(sample == 0, 1e-12, sample)


def logpdf_dirichlet_safe(val, alpha):
    if alpha.shape == (1,):
        return jnp.array([0.0])
    return genjax.dirichlet.logpdf(val, alpha)


dirichlet = genjax.exact_density(sample_dirichlet_safe, logpdf_dirichlet_safe)


def unwrap(x):
    """Unwrap `x` if it is a `Const`; otherwise return `x`."""
    if isinstance(x, Const):
        return x.val
    else:
        return x


class MyPytree(PythonicPytree):
    """
    Pytree base class with some extra bells and whistles, including:
        - supports self.replace(...) to functionally update fields
    """

    def replace(self: T, **kwargs) -> T:
        return replace(self, **kwargs)

    @staticmethod
    def eq(x, y):
        # See https://github.com/probcomp/genjax/issues/1441 for why
        # I didn't just override __eq__.
        # (Could get the __eq__ override to work with a bit more effort, however.)
        if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
            return False
        leaves1 = jax.tree_util.tree_leaves(x)
        leaves2 = jax.tree_util.tree_leaves(y)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    # The __getitem__ override is needed for GenJAX versions
    # prior to https://github.com/probcomp/genjax/pull/1440.
    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda v: v[idx], self)


def normalize(arr):
    return arr / jnp.sum(arr)


@Pytree.dataclass
class Intrinsics(Pytree):
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float
    image_height: int = Pytree.static()
    image_width: int = Pytree.static()

    def downscale(self, factor):
        return Intrinsics(
            self.fx / factor,
            self.fy / factor,
            self.cx / factor,
            self.cy / factor,
            self.near,
            self.far,
            self.image_height // factor,
            self.image_width // factor,
        )

    def crop(self, miny, maxy, minx, maxx):
        return Intrinsics(
            self.fx,
            self.fy,
            self.cx - minx,
            self.cy - miny,
            self.near,
            self.far,
            maxy - miny,
            maxx - minx,
        )


@Pytree.dataclass
class ImageWithIntrinsics(PythonicPytree):
    """
    An image paired with its intrinsics.

    (This supports cropping, downscaling, etc., with
    the intrinsics automatically being updated.)
    """

    image: jnp.ndarray  # (H, W, ...)
    intrinsics: Intrinsics

    def downscale(self, factor):
        return ImageWithIntrinsics(
            image=self.image[::factor, ::factor],
            intrinsics=self.intrinsics.downscale(factor),
        )

    def crop(self, miny, maxy, minx, maxx):
        return ImageWithIntrinsics(
            image=self.image[miny:maxy, minx:maxx],
            intrinsics=self.intrinsics.crop(miny, maxy, minx, maxx),
        )


def find_first_above(values, threshold):
    """
    Returns the first index where `values[idx] >= threshold`.
    If no such index exists, returns -1.
    """
    first = jnp.argmax(values >= threshold)
    return jnp.where(jnp.logical_and(first == 0, values[0] < threshold), -1, first)


### FloatFromDiscreteSet ###
@dataclass
class Domain:
    """
    Represents the domain of a :class:`FloatFromDiscreteSet`.
    """

    # JAX array of the values in this domain.
    # At runtime, this will live on the GPU, and
    # accessing `FloatFromDiscreteSet.value` will
    # index into this array.
    values: jnp.ndarray

    # A copy of the values on the CPU, for use at compile time.
    _numpy_values: np.ndarray

    def __init__(self, values):
        self.values = values
        self._numpy_values = np.array(values)

    @property
    def discrete_float_values(self):
        """
        A batched `FloatFromDiscreteSet` containing
        each element in this domain.
        """
        return jax.vmap(lambda idx: FloatFromDiscreteSet(idx=idx, domain=self))(
            jnp.arange(self.values.shape[0])
        )

    def first_value_above(self, val) -> "FloatFromDiscreteSet":
        """
        Return a `FloatFromDiscreteSet` for the smallest value
        greater than or equal `val` in the domain.

        If no such value exists, returns FloatFromDiscreteSet(-1, domain).
        """
        idx = find_first_above(self.values, val)
        return FloatFromDiscreteSet(idx=idx, domain=self)

    def __eq__(self, other):
        return bool(np.all(self._numpy_values == other._numpy_values))

    def __len__(self):
        return len(self.values)

    def __hash__(self):
        return hash(tuple(self._numpy_values))


@Pytree.dataclass
class FloatFromDiscreteSet(MyPytree):
    """
    Represents a floating point value that is one of a
    discrete set of possible floating point values.
    Use `.value` to get the represented value.
    Use `.idx` to get the index of the value in the set
    of possible values.

    ### Representation
    A `FloatFromDiscreteSet` object stores
    a reference to JAX array `domain` containing the set of
    all possible values this variable can take,
    and the index of the represented value in that array.

    ### Motivation
    When a probability distribution is represented
    as a probability vector over the domain of a
    `FloatFromDiscreteSet` variable, this index-based
    representation enables evaluating the PMF
    in O(1) time.  (Conversely, storing the float directly
    would require searching the domain for the float
    before the PMF could be evaluated, taking
    O(log n) time.)
    """

    idx: int
    domain: Domain = Pytree.static()

    @property
    def value(self):
        return self.domain.values[self.idx]

    @property
    def shape(self):
        return self.idx.shape

    def tile(self, *tile_args, **tile_kwargs):
        return FloatFromDiscreteSet(
            idx=jnp.tile(self.idx, *tile_args, **tile_kwargs), domain=self.domain
        )

    def __eq__(self, other):
        return self.domain == other.domain and jnp.all(
            jnp.array(self.idx) == jnp.array(other.idx)
        )


@genjax.Pytree.dataclass
class IndexSpaceUniform(genjax.ExactDensity):
    """
    A distribution over `FloatFromDiscreteSet` objects,
    where the distribution over indices into the `FloatFromDiscreteSet`
    `Domain` is uniform.

    Distribution arguments:
    - `domain`: a `Domain`.

    Distribution support:
    - The support is the set of all `FloatFromDiscreteSet` objects
        with the given `Domain`.
    """

    def sample(self, key, domain: Domain):
        idx = jax.random.randint(key, (), 0, len(domain))
        return FloatFromDiscreteSet(idx=idx, domain=domain)

    def logpdf(self, x: FloatFromDiscreteSet, domain):
        assert x.domain == domain
        return -jnp.log(len(domain))

    @property
    def __doc__(self):
        return IndexSpaceUniform.__doc__


index_space_uniform = IndexSpaceUniform()


### Pytree indexing utils ###
def mywhere(b, x, y):
    """
    Like jnp.where(b, x, y), but can handle cases like
    b.shape = (N,) while x.shape = y.shape = (N, M, ...).
    """
    assert len(x.shape) == len(y.shape)
    if len(b.shape) == len(x.shape):
        return jnp.where(b, x, y)
    else:
        return jnp.where(
            b[:, *(None for _ in range(len(x.shape) - len(b.shape)))], x, y
        )


def replace_slots_in_seq(
    seq: MyPytree,  # Batched, (T, N, ...)
    replacements: MyPytree,  # Batched, (N, ...)
    do_replace: jnp.ndarray,  # (T, N)
):
    return jax.tree.map(
        lambda s, r: jax.vmap(
            lambda x, rep, do_rep: mywhere(do_rep, rep, x), in_axes=(0, None, 0)
        )(s, r, do_replace),
        seq,
        replacements,
    )


def uniformly_replace_slots_in_seq(
    seq: MyPytree,  # Batched, (T, N, ...)
    replacements: MyPytree,  # Batched, (N, ...)
    do_replace: jnp.ndarray,  # (N,)
):
    T = len(seq)
    return replace_slots_in_seq(seq, replacements, jnp.tile(do_replace, (T, 1)))


### Manipulating traces


# Extract relevant info for the update from the trace
def markov_for_xy_mean_from_trace(trace):
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "xy"]
    n_clusters = trace.get_args()[0].n_blobs
    prior_mean = trace.get_args()[0].mu_xy
    current_means = trace.get_choices()["blob_model", "xy_mean"]  # shape (N,2)
    prior_variance = trace.get_choices()["blob_model", "sigma_xy"]
    obs_variance = trace.get_args()[0].sigma_xy

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        prior_variance,
        obs_variance,
    )


# Count the number of points per cluster
def category_count(datapoint_indexes, n_clusters):
    return jnp.bincount(
        datapoint_indexes,
        length=n_clusters,
        minlength=n_clusters,
    )


# Update the trace with new xy_mean
def update_trace_with_xy_mean(key, trace, new_means):
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        key, C["blob_model", "xy_mean"].set(new_means), argdiffs
    )
    return new_trace

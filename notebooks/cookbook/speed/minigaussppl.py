from dataclasses import dataclass
from functools import cached_property
from typing import Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

# Some helpers
Matrix = Vector = jax.Array
Real = float | jax.Array


def Maybe(Type):
    return Type | type(None)


def mat_square(mat):
    return mat.T @ mat


def split_mat(mat, idxs1, idxs2):
    return (
        mat[*jnp.meshgrid(idxs1, idxs1, indexing="ij")],
        mat[*jnp.meshgrid(idxs1, idxs2, indexing="ij")],
        mat[*jnp.meshgrid(idxs2, idxs1, indexing="ij")],
        mat[*jnp.meshgrid(idxs2, idxs2, indexing="ij")],
    )


@dataclass
class JaxKey:
    _key: jax.Array

    def __post_init__(self):
        def key_gen():
            cur_key = self._key
            while True:
                yield cur_key
                cur_key = jax.random.split(cur_key, 1).flatten()

        self.key_gen = key_gen()

    def __enter__(self):
        return self

    def __exit__(*args):
        if all(arg is None for arg in args):
            return True
        return False

    @property
    def key(self):
        return next(self.key_gen)

    def __call__(self):
        return self.key


@register_pytree_node_class
@dataclass
class LinearGaussianKernel:
    tran: Matrix
    cov: Matrix

    def at(self, x: Vector) -> "Gaussian":
        assert x.shape == (self.tran.shape[1],)
        return Gaussian(self.tran @ x, self.cov)

    def tree_flatten(self):
        aux_data = None
        children = (self.tran, self.cov)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        _ = aux_data
        tran, cov = children
        return cls(tran, cov)


@register_pytree_node_class
@dataclass
class Gaussian:
    mean: Vector
    cov: Matrix

    @cached_property
    def dim(self):
        return len(self.mean)

    @cached_property
    def is_degenerate(self):
        return jnp.isclose(jnp.linalg.det(self.cov), 0)

    def sample(self, key: jax.Array) -> Vector:
        eigvals, eigvecs = jnp.linalg.eigh(self.cov)
        A = eigvecs @ jnp.diag(jnp.sqrt(eigvals))
        return A @ jax.random.normal(key, shape=self.mean.shape) + self.mean

    def logpdf(self, x: Vector) -> Maybe(Real):
        return jax.lax.cond(
            self.is_degenerate,
            lambda: jnp.nan,
            lambda: jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov),
        )

    def condition(self, x: Vector, idxs: Vector) -> Self:
        # https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
        assert x.shape == idxs.shape
        # idxs_ = complement of `idxs`: remaining idxs
        idxs_ = jnp.delete(jnp.arange(self.dim), idxs, assume_unique_indices=True)
        cov11, cov12, cov21, cov22 = split_mat(self.cov, idxs_, idxs)
        mean1, mean2 = self.mean[idxs_], self.mean[idxs]
        cov22_inv = jnp.linalg.pinv(cov22)
        cond_mean = mean1 + cov12 @ cov22_inv @ (x - mean2)
        cond_cov = cov11 - cov12 @ cov22_inv @ cov21
        return Gaussian(cond_mean, cond_cov)

    def join(self, k: LinearGaussianKernel) -> Self:
        assert self.dim == k.tran.shape[1]
        joint_mean = jnp.hstack([self.mean, k.tran @ self.mean])
        joint_cov = jnp.block([
            [self.cov, self.cov @ k.tran.T],
            [k.tran @ self.cov, k.cov + k.tran @ self.cov @ k.tran.T],
        ])
        return Gaussian(joint_mean, joint_cov)

    def tree_flatten(self):
        aux_data = None
        children = (self.mean, self.cov)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        _ = aux_data
        mean, cov = children
        return cls(mean, cov)

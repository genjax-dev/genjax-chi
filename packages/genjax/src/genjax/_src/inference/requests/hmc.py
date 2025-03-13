# Copyright 2024 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import grad
from jax.lax import scan
from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    EditRequest,
    Retdiff,
    Score,
    Selection,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.requests import DiffAnnotate
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    IntArray,
    PRNGKey,
    static_check_supports_grad,
)

tfd = tfp.distributions


# Pytree manipulation utilities -- these handle unzipping Pytrees into
# differentiable and non-diff pieces, and then also zipping them back up.
def grad_tree_unzip(tree: ChoiceMap) -> tuple[ChoiceMap, ChoiceMap]:
    grad_tree = jtu.tree_map(
        lambda v: v if static_check_supports_grad(v) else None, tree
    )
    nongrad_tree = jtu.tree_map(
        lambda v: v if not static_check_supports_grad(v) else None, tree
    )
    return grad_tree, nongrad_tree


def grad_tree_zip(
    grad_tree: ChoiceMap,
    nongrad_tree: ChoiceMap,
) -> ChoiceMap:
    return jtu.tree_map(
        lambda v1, v2: v1 if v1 is not None else v2, grad_tree, nongrad_tree
    )


# Compute the gradient of a selection of random choices
# in a trace -- uses `GenerativeFunction.assess`.
def selection_gradient(
    selection: Selection,
    trace: Trace[Any],
    argdiffs: Argdiffs,
) -> tuple[ChoiceMap, ChoiceMap]:
    chm = trace.get_choices()
    filtered = chm.filter(selection)
    complement = chm.filter(~selection)
    grad_tree, nongrad_tree = grad_tree_unzip(filtered)
    gen_fn = trace.get_gen_fn()

    def differentiable_assess(grad_tree):
        zipped = grad_tree_zip(grad_tree, nongrad_tree)
        full_choices = zipped.merge(complement)
        weight, _ = gen_fn.assess(
            full_choices,
            Diff.tree_primal(argdiffs),
        )
        return weight

    return grad_tree_zip(grad_tree, nongrad_tree), jtu.tree_map(
        lambda v1, v2: v1
        if v1 is not None
        else jnp.zeros_like(jnp.array(v2, copy=False)),
        grad(differentiable_assess)(grad_tree),
        nongrad_tree,
    )


# Utilities for momenta sampling and score evaluation.
def normal_sample(key: PRNGKey, shape) -> FloatArray:
    return tfd.Normal(jnp.zeros(shape), 1.0).sample(seed=key)


def normal_score(v) -> Score:
    score = tfd.Normal(0.0, 1.0).log_prob(v)
    if score.shape:
        return jnp.sum(score)
    else:
        return score


def assess_momenta(momenta, mul=1.0):
    return jnp.sum(
        jnp.array(
            jtu.tree_leaves(jtu.tree_map(lambda v: normal_score(mul * v), momenta))
        )
    )


def sample_momenta(key, choice_gradients):
    total_length = len(jtu.tree_leaves(choice_gradients))
    int_seeds = jnp.arange(total_length)
    int_seed_tree = jtu.tree_unflatten(jtu.tree_structure(choice_gradients), int_seeds)
    momenta_tree = jtu.tree_map(
        lambda v, int_seed: normal_sample(jrand.fold_in(key, int_seed), v.shape),
        choice_gradients,
        int_seed_tree,
    )
    momenta_score = assess_momenta(momenta_tree)
    return momenta_tree, momenta_score


#######
# HMC #
#######


@Pytree.dataclass(match_args=True)
class HMC(EditRequest):
    """
    Apply a Hamiltonian Monte Carlo (HMC) update that proposes new values for the selected addresses,
    returning the new trace, and a weight which is equal to the alpha accept-reject ratio computation for HMC.

    Hamilton's equations are numerically integrated using leapfrog integration with step size `eps` for `L` steps.
    See equations (5.18)-(5.20) of Neal (2011).

    # References
    Neal, Radford M. (2011), "MCMC Using Hamiltonian Dynamics", Handbook of Markov Chain Monte Carlo,
    pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf
    """

    selection: Selection
    eps: FloatArray
    L: int = Pytree.static(default=10)

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[Any],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[Any], Weight, Retdiff[Any], "EditRequest"]:
        # Just a conservative restriction, for now.
        assert Diff.static_check_no_change(argdiffs)

        original_model_score = tr.get_score()
        values, gradients = selection_gradient(self.selection, tr, argdiffs)
        key, sub_key = jrand.split(key)
        momenta, original_momenta_score = sample_momenta(sub_key, gradients)

        def kernel(
            carry: tuple[Trace[Any], ChoiceMap, ChoiceMap, ChoiceMap],
            scanned_in: IntArray,
        ) -> tuple[tuple[Trace[Any], ChoiceMap, ChoiceMap, ChoiceMap], Retdiff[Any]]:
            trace, values, gradient, momenta = carry
            int_seed = scanned_in
            momenta = jtu.tree_map(
                lambda v, g: v + (self.eps / 2) * g, momenta, gradient
            )
            values = jtu.tree_map(lambda v, m: v + self.eps * m, values, momenta)
            new_key = jrand.fold_in(key, int_seed)
            new_trace, _, retdiff, _ = Update(values).edit(new_key, trace, argdiffs)
            values, gradients = selection_gradient(self.selection, new_trace, argdiffs)
            momenta = jtu.tree_map(
                lambda v, g: v + (self.eps / 2) * g, momenta, gradients
            )
            return (new_trace, values, gradient, momenta), retdiff

        int_seeds = jnp.arange(self.L) + 1
        (final_trace, _, _, final_momenta), retdiffs = scan(
            kernel,
            (tr, values, gradients, momenta),
            int_seeds,
            length=self.L,
        )

        final_model_score = final_trace.get_score()
        final_momenta_score = assess_momenta(final_momenta, mul=-1.0)
        alpha = (
            final_model_score
            - original_model_score
            + final_momenta_score
            - original_momenta_score
        )
        # Grab the last retdiff.
        retdiff = jtu.tree_map(lambda v: v[-1], retdiffs)
        return (
            final_trace,
            alpha,
            retdiff,
            HMC(self.selection, self.eps, self.L),
        )


def SafeHMC(
    selection: Selection,
    eps: FloatArray,
    L: int = 10,
) -> DiffAnnotate[HMC]:
    def retdiff_assertion(retdiff: Retdiff[Any]):
        assert Diff.static_check_no_change(retdiff)
        return retdiff

    return HMC(selection, eps, L).map(retdiff_assertion)

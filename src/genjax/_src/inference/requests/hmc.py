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


def grad_tree_unzip(tree):
    grad_tree = jtu.tree_map(
        lambda v: v if static_check_supports_grad(v) else None, tree
    )
    nongrad_tree = jtu.tree_map(
        lambda v: v if not static_check_supports_grad(v) else None, tree
    )
    return grad_tree, nongrad_tree


def grad_tree_zip(grad_tree, nongrad_tree):
    return jtu.tree_map(lambda v1, v2: v1 if v1 else v2, grad_tree, nongrad_tree)


def selection_gradient(selection, trace, argdiffs):
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

    return grad_tree, grad(differentiable_assess)(grad_tree)


def normal_sample(key, shape) -> FloatArray:
    return tfp.Normal(jnp.zeros(shape), 1.0).sample(key)


def normal_score(v) -> Score:
    score = tfp.Normal(0.0, 1.0).logprob(v)
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


@Pytree.dataclass(match_args=True)
class HMC(EditRequest):
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
        choice_values, choice_gradients = selection_gradient(
            self.selection, tr, argdiffs
        )
        key, sub_key = jrand.split(key)
        momenta, original_momenta_score = sample_momenta(sub_key, choice_gradients)

        def kernel(
            carry: tuple[Trace[Any], ChoiceMap, ChoiceMap, ChoiceMap],
            scanned_in: IntArray,
        ) -> tuple[tuple[Trace[Any], ChoiceMap, ChoiceMap, ChoiceMap], None]:
            trace, values, gradient, momenta = carry
            int_seed = scanned_in
            momenta = jtu.tree_map(
                lambda v, g: v + (self.eps / 2) * g, momenta, gradient
            )
            values = jtu.tree_map(lambda v, m: v + self.eps * m, values, momenta)
            new_key = jrand.fold_in(key, int_seed)
            new_trace, _, retdiff, _ = Update(values).edit(new_key, trace, argdiffs)
            values, gradient = selection_gradient(self.selection, new_trace, argdiffs)
            momenta = jtu.tree_map(
                lambda v, g: v + (self.eps / 2) * g, momenta, gradient
            )
            return (new_trace, values, gradient, momenta), retdiff

        (final_trace, _, _, final_momenta), retdiffs = scan(
            kernel, (tr, choice_values, choice_gradients, momenta), None, length=self.L
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

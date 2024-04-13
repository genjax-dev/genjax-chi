import jax

import genjax


@genjax.unfold_combinator(max_length=100)
@genjax.static_gen_fn
def random_walk(a):
    x = genjax.normal(a, 1.0) @ "x"
    return x


key = jax.random.PRNGKey(0)
tr = random_walk.simulate(key, (20, 0.0))

tr2_ = random_walk.update(
    key,
    tr,
    genjax.EmptyChoice(),
    (
        genjax.Diff.tree_diff(21, genjax.UnknownChange),
        genjax.Diff.tree_diff(0.0, genjax.NoChange),
    ),
)

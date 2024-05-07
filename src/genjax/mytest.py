import jax.numpy as jnp

import genjax


def selection_from_dict(dict):
    trie = genjax.Trie()
    for k, v in dict.items():
        assert isinstance(v, genjax.Selection)
        trie = trie.trie_insert(k, v)
    return genjax.HierarchicalSelection(trie)


s = selection_from_dict(
    {
        "particle_poses": genjax.indexed_select(
            jnp.array(range(10)), genjax.select("pose")
        ),
        "cluster_assignments": genjax.AllSelection(),
        "chain": selection_from_dict(
            {
                "cluster_poses": genjax.indexed_select(
                    jnp.array(range(1, 10)), genjax.AllSelection()
                ),
                "observed_particle_poses": genjax.indexed_select(
                    jnp.array(range(1, 10)), genjax.AllSelection()
                ),
            }
        ),
    }
)

# Changelog

## [unreleased]

- `genjax.choice_map`, when given a `dict` argument, as in `genjax.choice_map({"x": {"y": v}})`,
  used to allow address tuples as keys, as in `genjax.choice_map({("x", "y"): v}). This is no
  longer allowed: if you want to use a Python dictionary to provide the structure of a choice
  map, just nest it as in the former example; there is no more implicit "key extension."

  It remains the case that addresses of choices in such maps are understood to be tuples,
  however, and there are other interfaces such as `choice_map.insert` where such a tuple can
  be used to indicate an interior part of a choice map.

## [v0.1.0]

- First numbered release! We began maintaining CHANGELOG entries from this point
  forward.

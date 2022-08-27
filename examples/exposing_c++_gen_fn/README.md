# Calling into C++ generative functions

This is a minimum working example of exposing C++ generative functions to `GenJAX` utilizing the XLA `custom_call` primitive.

[There's associated documentation on the `GenJAX` documentation site](https://probcomp.github.io/genjax/genjax/c_interface.html).

This sub-directory is built by CI/CD and checked as part of the standard set of tests - so the usability of its functionality should be guaranteed by CI/CD. Generally, if you wish to port over your own generative function implementations - we recommend copying this directory.

Building the directory with `pybind11` and `cmake` is orchestrated via `pip`:

```
pip install .
```

This will expose the built `pybind11` module to your Python environment:

```
python
> import extern_gen_fn
```

Allowing you to utilize the registered module methods defined in `lib/main.cpp`.

Exposing C++ generative functions
=================================

While GenJAX is fast on GPUs courtesy of JAX, but sometimes we want lower level control over the implementation of our generative functions -- or we want to emit code which is optimized for CPU devices (so we wish to leverage modern CPU compiler toolchains like LLVM). 

Fortunately, we can expose generative functions written in C or C++ by utilizing an XLA primitive called `custom_call`_. In this note, I'll walk through the process of exposing a generative function written in C++ using `GenTL`_ to GenJAX. This note is based on `this excellent note`_ by Dan Foreman-Mackey.

A new module
------------

To expose the functionality we need, we'll setup a new `pybind11` module built using using `CMake`_. I've done so `in this directory of GenJAX`_.

We can use any supporting infrastructure we want, including header-only libraries like `GenTL`_ which provide tooling to define generative functions in C++. In `include/gen_fn.h`_ - there's a sketch of our C++ generative function, and `lib/gen_fn.cpp`_ provides an implementation of the templated methods.

.. _custom_call: https://www.tensorflow.org/xla/custom_call
.. _GenTL: https://github.com/OpenGen/GenTL
.. _this excellent note: https://dfm.io/posts/extending-jax/
.. _CMake: https://cmake.org/
.. _in this directory of GenJAX: https://github.com/probcomp/genjax/tree/main/examples/exposing_c++_gen_fn
.. _include/gen_fn.h: https://github.com/probcomp/genjax/blob/main/examples/exposing_c%2B%2B_gen_fn/include/gen_fn.h
.. _lib/gen_fn.cpp: https://github.com/probcomp/genjax/blob/main/examples/exposing_c%2B%2B_gen_fn/lib/gen_fn.cpp

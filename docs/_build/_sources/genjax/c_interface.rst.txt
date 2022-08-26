Calling into C++ generative functions
=====================================

While GenJAX is fast on GPUs courtesy of JAX, sometimes we want lower level control over the implementation of our generative functions -- or we want to emit code which is optimized for CPU devices (so we wish to leverage modern CPU compiler toolchains like LLVM). 

Fortunately, we can expose generative functions written in C or C++ by utilizing an XLA primitive called `custom_call`_. In this note, I'll walk through the process of exposing a generative function written in C++ using `GenTL`_ to GenJAX. This note is based on `this excellent note`_ by Dan Foreman-Mackey.

A new module
------------

To expose the functionality we need, we'll setup a new Python module with 
`CMake`_ build support. I've done so `in this directory of GenJAX`_.

.. _custom_call: https://www.tensorflow.org/xla/custom_call
.. _GenTL: https://github.com/OpenGen/GenTL
.. _this excellent note: https://dfm.io/posts/extending-jax/
.. _CMake: https://cmake.org/
.. _in this directory of GenJAX: https://cmake.org/

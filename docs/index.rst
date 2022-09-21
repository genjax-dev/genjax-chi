.. genjax documentation master file, created by
   sphinx-quickstart on Fri Aug 26 07:06:27 2022.

Index
=====

`GenJAX`_ is a probabilistic programming framework constructed by combining the 
concepts of `Gen`_ with the hardware accelerator compilation capabilities 
of `JAX`_.

It exposes the ability to programmatically construct *generative functions* 
(c.f. :doc:`genjax/gen_fn`), computational objects which represent measures 
over structured sample spaces. These objects also expose a concise interface 
for expressing differentiable programming and Monte Carlo inference algorithms.
These objects are given a precise mathematical formulation in `Marco Cusumano-Towner's
PhD thesis`_.

  If you're new to `Gen`_ and probabilistic programming in general, you'll 
  likely want to start with :doc:`genjax/tour`.

  If you don't mind perusing carefully crafted documentation 
  (albeit in another language), you might also enjoy the `Gen`_
  Julia documentation.

.. toctree::
   :maxdepth: 2

   genjax/tour
   genjax/gen_fn
   genjax/interface
   genjax/combinators
   genjax/diff_jl
   genjax/c_interface
   genjax/wasm_interface

.. _GenJAX: https://github.com/probcomp/genjax
.. _Gen: https://www.gen.dev
.. _JAX: https://github.com/google/jax
.. _Marco Cusumano-Towner's PhD thesis: https://www.mct.dev/assets/mct-thesis.pdf

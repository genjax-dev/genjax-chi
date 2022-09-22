.. genjax documentation master file, created by
   sphinx-quickstart on Fri Aug 26 07:06:27 2022.

=====
Index
=====

`GenJAX`_ is system for probabilistic programming constructed by combining the concepts of `Gen`_ with the hardware accelerator compilation capabilities of `JAX`_.

GenJAX exposes the ability to programmatically construct and manipulate *generative functions* 
(c.f. :doc:`genjax/gen_fn`): computational objects which represent probability measures 
over structured sample spaces. 

These objects also expose a concise interface 
for expressing differentiable programming and Monte Carlo inference algorithms. A precise mathematical formulation is given in `Marco Cusumano-Towner's PhD thesis`_.

.. admonition:: Novice
  
  If you're new to `Gen`_ (or probabilistic programming in general), 
  you'll likely want to start with :doc:`genjax/tour`.

  If you don't mind perusing carefully crafted documentation 
  (albeit in another language), you might also enjoy the `Gen.jl`_
  Julia documentation.

.. toctree::
   :hidden:
   :caption: Getting started
   :maxdepth: 1

   genjax/tour
   genjax/gen_fn
   genjax/interface
   genjax/diff_jl
  
.. toctree::
   :hidden:
   :caption: Modeling and inference
   
   genjax/combinators/combinators
   genjax/inference/inference

.. toctree::
   :hidden:
   :caption: Experimental

   genjax/experimental/diagnostics
   genjax/experimental/trace_types
   genjax/experimental/prox

.. toctree::
   :hidden:
   :caption: Foreign function interfaces
   
   genjax/c_interface
   genjax/wasm_interface

.. _GenJAX: https://github.com/probcomp/genjax
.. _Gen: https://www.gen.dev
.. _Gen.jl: https://www.gen.dev/stable/
.. _JAX: https://github.com/google/jax
.. _Marco Cusumano-Towner's PhD thesis: https://www.mct.dev/assets/mct-thesis.pdf

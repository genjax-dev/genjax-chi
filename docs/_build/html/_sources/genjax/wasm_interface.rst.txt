From GenJAX to WebAssembly
==========================

Building a framework for generative functions on top of JAX allows us to take
advantage of deployment capabilities built into JAX proper. One of these
opportunities is the ability to convert :code:`jax.jit` capable Python programs
into :code:`tf.Module` instances - ready for deployment via 
:code:`TensorFlow.js` to WebAssembly modules. This is one path to support web-native usage of generative functions.

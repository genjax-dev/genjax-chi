# The runtime debugger

**Module documentation**
::: genjax._src.core.runtime_debugger
options:
show_root_heading: true

GenJAX features a runtime (1) debugging system implemented via JAX transformations on stateful functions.
{ .annotate }

1.  Here, _runtime_ is used in contrast to JAX tracing time (which is akin to "compile time"). The runtime debugger can be used to inspect values of functions which JAX can successfully trace through.

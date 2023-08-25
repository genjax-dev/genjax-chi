# The runtime debugger

GenJAX features a runtime (1) debugging system implemented via JAX transformations on stateful functions. The debugger will not detect static errors or tracing errors, but can be used to inspect the values flowing through your code.
{ .annotate }

1.  Here, _runtime_ is used in contrast to JAX tracing time (which is akin to "compile time"). The runtime debugger can be used to inspect values of functions which JAX can successfully trace through.

::: genjax.core.runtime_debugger
    options:
      show_root_heading: true
      members:
        - record_value
        - record_call
        - record
        - tag
        - pull

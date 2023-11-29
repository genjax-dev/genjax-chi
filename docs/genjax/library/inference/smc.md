# Sequential Monte Carlo

## Trace translators

In Gen, generative functions define probability distributions on traces. A _trace translator_ converts one space of traces to another space of traces, and can be used as building blocks for inference programs that utilize multiple model representations, deterministic transformations of choice structure, and more.

In GenJAX, trace translators are a critical ingredient for SMC. GenJAX carefully restricts usage of dynamism(1) in its modeling languages, and trace translators provide a way to utilize dynamism to concisely express inference.
{ .annotate }

1. _Dynamism_ means runtime-determined values.

## SMC extension steps via `ExtendingTraceTranslator`

## SMC moves with probabilistic program proposals via `TraceKernelTraceTranslator`

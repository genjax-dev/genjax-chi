# Inference
In probabilistic programming, inference refers to the construction of normalized conditional distributions. 

In GenJAX, the inference stack is based on an implementation of the language described in [Probabilistic Programming with Stochastic Probabilities](https://dl.acm.org/doi/abs/10.1145/3591290) (GenSP). The core concepts include data types which represent posterior inference problems (`Target`), inference algorithms (`InferenceAlgorithm`), and distribution objects which expose sampling and log density _estimation_. 

The concepts from GenSP allow GenJAX to support new modeling constructs like `Marginal`, which denotes marginalization over random variables. This object exposes sampling and density _estimation_ interfaces which cohere with Gen's existing estimator semantics.

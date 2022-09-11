# Gen Inference and Learning Template Library (GenTL)   {#mainpage}

Applying probabilistic modeling, inference, and learning is made easier by a separation of concerns between the definition of a probabilistic model and the implementation of inference and learning algorithms that operate on the model.
One approach to achieving this separation is to define an interface that a probabilistic model should implement, and then write inference and learning algorithms that only interact with the model through that interface.
The GenTL library provides high-level building blocks for implementing inference and learning algorithms that can be applied to probabilistic models that implement a specific interface.
GenTL includes template functions and template classes whose template parameters are user-defined types that implement the interface for the user's probabilistic model.
Different features of GenTL require different (but overlapping) functions to be defined for these user-defined types, and it is not often necessary to implement all of the functionality required by all GenTL features.

GenTL is designed to be used with either hand-coded user-defined types for a probabilistic model, or types that use probabilistic programming languages to automatically implement the functionality in the interface.
Implementing your model in a probabilistic programming language that is compatible with GenTL may be an appropriate first step in the development process, because iterating on the model is faster, there is less code required, and iterating on the inference algorithm is easier since the probabilistic programming language will (ideally) implement the full interface required by any inference and learning features within GenTL.
After implementing the model using a probabilistic programming language, you can optimize performance by re-implementing the types by hand, without having to rewrite the GenTL-based inference and learning code.
It is also possible to incrementally migrate parts of the model from a probabilistic programming based implementation to a custom implementation, if the probabilistic programming language can invoke 'foreign' probabilistic programs that implement GenTL's interfaces.

## Distributions, traces, and generative functions

There are three main types of objects that together constitute a probabilistic model that can be used with GenTL.

- Objects that represent a joint probability distribution over some random variables (GenTL also can be used with distributions over measure spaces with variable-dimension). These objects are called **distributions**.

- Objects that represent a sample from such a probability distribution. These objects are called **traces**.

- Objects that are callable, and when called return a distribution. These objects are called **generative functions**.

There are also other auxiliary types of objects that are used in the interface:

- Objects that represent changes to a value (**changes**).

- Objects that represent values of some subset of random variables in a random sample (**choice buffers**).

- Objects that contain values for learnable parameters of a distribution (**parameter stores**).

- Objects that contain gradient accumulators for learnable parameters of a distribution (**gradient accumulators**).

- Objects that represent selections of some part of a (joint) sample value (**selections**).

### Distribution objects

Distribution objects may implement two different techniques for sampling traces, implemented in the interface functions called `simulate` and `generate`. Briefly:

- The [`simulate`](#simulate) member function samples a trace exactly from the (joint) distribution.

- The [`generate`](#generate) member function samples a trace from an alternative distribution defined by the distribution object (optionally subject to constraints on the sample) and also returns an importance weight relating the joint distribution to the sampling distribution.  The constraints on the sample are represented by a **choice buffer**.

Note that importance weights, densities, and other ratios of densities, are always returned in log-space.

### Trace objects

Trace objects always record a distribution that they are associated with.

- The [`score`](#score) member function reports the joint density of the trace's sample under the trace's distribution.

- The [`update`](#update) member function can be used to change the distribution that a trace is associated with. It also reports the ratio of densities between a new and old trace. If the sample value for the original distribution does not uniquely specify a sample value for the new distribution, the function may perform sampling, and return an importance weight instead. The change to a distribution is represented by a change object.

- The `update` member function can also be used to modify the sample value stored in the trace so that it satisfies given constraints (a choice buffer), together with or independently of a change to the distribution.

- Trace objects may also store a value that is separate from their sample value, called the **return value**, which is accessible via the [`return_value`](#return_value) member function. The return value is a function of the sample value. After a call to `update`, the trace can be queried for a change object that represents the change to the return value (if any), via the [`return_value_change`](#return_value_change) member function.

- The effect of an `update` can be undone with the [`revert`](#revert) member function, if the `save` flag was set in the `update` options.

- A trace can be copied using the [`fork`](#fork) member function, which returns a unique pointer to a trace that is identical but independent from `this` trace from this point forward. A trace can also be forked from another trace of the same type, overwriting its contents with a copy of the contents of the other trace, via the [`fork_from`](#fork_from) member function.

- Trace objects support taking gradients of the log-density with respect to some component(s) of the random sample, via the [`choice_gradient`](#choice_gradient) member function, which optionally accepts a return value gradient to support compositional implementations based on reverse-mode automatic differentiation.

- Trace objects support taking gradients of the log-density with respect to learnable parameters, via the [`parameter_gradient`](#parameter_gradient) member function, which optionally accepts a return value gradient to support compositional implementations based on reverse-mode automatic differentiation, and optionally scales the gradients by some number before accumulating them into a provided gradient accumulator object.

### Generative function objects

Generative function objects are callable objects that, when called on a single argument (called the **input**), return a distribution object.

Generative functions are compositional building blocks of generative models that behave like mathematical functions (from their input to the return value of the trace that they generate), but they are stochastic (due to the random value of their sample).
If you want your generative model to be resuable as a building block within other generative models, then you should implement a generative function for it.
For models that are represented as generative functions, there is a dependency between the change objects that can be passed to the `update` function of a trace, and the set of valid inputs to the generative function from which the trace originated:
`update` must accept change objects that encode a change of the input to the generative function to any valid input value.

<div id="interface"></div>
## Interface documentation

This section describes the member functions that are expected by GenTL for user-defined types that represent distributions, traces, generative functions, and the other auxiliary types.
The requirements that GenTL places on user-defined types are defined in include/gentl/concepts.h as C++20 [concepts](https://en.wikipedia.org/wiki/Concepts_(C%2B%2B)) (they will not be checked by the compiler if the compiler does not support Concepts, but they still serve as documentation).

### Semantics

TODO: document encapsulated randomness

$$
   \\def\\ss{{\\boldsymbol{\\sigma}}}
   \\def\\uu{{\\boldsymbol{\\upsilon}}}
   \\def\\tt{{\\boldsymbol{\\tau}}}
$$

The semantics below assumes that samples are mathematical objects called **choice dictionaries**.
(Note that some of the features in GenTL do not require that samples have this structure, but we document it here for concreteness.)
A choice dictionary is a finite map from the **address** of a random choice to its **value**.
See [Chapter 2 and Chapter 4 of this PhD thesis](https://www.mct.dev/assets/mct-thesis.pdf) for a mathematical definition of generative functions and traces over choice dictionaries.
We will assume that all random choices are discrete. 
We use the notation from that thesis below
(we will denote choice dictionaries using the variables \f$\tt\f$, \f$\uu\f$, and \f$\ss\f$,
and a probability distribution on choice dictionaries is denoted \f$p\f$, with probability density (mass) for dictionary \f$\tt\f$ denoted \f$p(\tt)\f$).
Also see the [Involutive MCMC paper](https://arxiv.org/abs/2007.09871) for a formulation of choice dictionaries.

### Distribution interface member functions

Each distribution object is associated with some trace type, which we will call `Trace` in the documentation that follows.
Note that GenTL does not contain any abstract class `Trace`.
Similarly, the other types used below, including `RNG`, `ParameterStore`, `ChoiceBuffer`, `GradientAccumulator`, and `Selection`, do not refer to abstract classes.
GenTL functions use template parameters for all of these types.
With the exception of `RNG` these types are expected to be user-defined.

<h4 id="simulate">simulate</h4>

```
std::unique_ptr<Trace> simulate(RNG& rng, ParameterStore& params, const SimulateOptions& options) const;
```

Sample \f$ \tt \sim p \f$ and return a new trace object containing \f$\tt\f$.

There is also an in-place variant that over-writes an existing trace object:

```
void simulate(RNG& rng, ParameterStore& params, const SimulateOptions& options, Trace& trace) const;
```

Also see gentl::SimulateOptions.

<h4 id="generate">generate</h4>

```
std::pair<std::unique_ptr<Trace>, double> simulate(RNG& rng, ParameterStore& params, const ChoiceBuffer& constraint, const GenerateOptions& options) const;
```

Given constraint \f$ \ss \f$, sample \f$ \uu\sim q(\cdot; \ss) \f$ and return a new trace object containing \f$\tt := \ss \oplus \uu\f$.

There is also an in-place variant that over-writes an existing trace object:

```
double generate(RNG& rng, ParameterStore& params, const ChoiceBuffer& constraint, const GenerateOptions& options, Trace& trace) const;
```

Also see gentl::GenerateOptions.

### Trace interface member functions

<h4 id="score">score</h4>

```
double score() const;
```

Return \f$ \log p(\tt) \f$.

<h4 id="update">update</h4>

```
double update(RNG& rng, const Change& change, const ChoiceBuffer& constraint, const UpdateOptions& options);
```

TODO: document math

Also see gentl::UpdateOptions.

The MCMC functionality in gentl::mcmc requires that an overload of `update` is provided for `Change` equal to gentl::change::NoChange`.

<h4 id="backward_constraints">backward_constraints</h4>

TODO: this should probably be moved to be an additional return value from `update`, since it's return type may depend on the type of the constraints passed to `update`.
Users should be permitted to implement multiple variants of `update` that accept different choice buffer types and return different choice buffer types as backward constraints.

```
const ChoiceBuffer& backward_constraints();
```

TODO: document math (this corresponds to the 'discard' object in Gen.jl).

The reference is invalidated by a subsequent call to `update` or `revert`.


<h4 id="return_value">return_value</h4>

```
const Return& return_value() const;
```

Return \f$ f(\tt) \f$.
The reference is invalidated by a subsequent call to `update` or `revert`.

<h4 id="return_value_change">return_value_change</h4>

```
const ReturnChange& return_value_change() const;
```

The reference is invalidated by a subsequent call to `update` or `revert`.

<h4 id="revert">revert</h4>

```
void revert();
```

Undo an `update` for which the `save` flag option was set.
Once reverted, a trace cannot be reverted again until after another call to `update`.

<h4 id="fork">fork</h4>

```
std::unique_ptr<Trace> fork();
```

Return a copy of the trace.
Future updates to `this` do not effect the returned trace, and vice versa.

<h4 id="fork_from">fork_from</h4>

```
void fork_from(const Trace& other);
```

Overwrite the contest of the `this` to contain a copy of the contents of `other`.
Future updates to `this` do not effect the returned trace, and vice versa.

<h4 id="choice_gradient">choice_gradient</h4>

```
const ChoiceBuffer& choice_gradient(const Selection& selection)
```

Return a reference to a choice buffer containing the gradient of \f$\log p(\tt)\f$ with respect to the values of the selected choices.
The reference is invalidated by a subsequent call to `update` or `revert`.


```
InputGradient choice_gradient(const Selection& selection, const RetGradient&)
```

Compositional variant designed for use with reverse-mode automatic differentiation.


<h4 id="parameter_gradient">parameter_gradient</h4>

```
void parameter_gradient(GradientAccumulator& accum, double scaler=1.0)
```

Increment the gradient accumulator by \f$\alpha \cdot \nabla_{\theta} \log p(\tt)\f$<!--\_--> where \f$\alpha\f$ is the value of `scaler`.

### Generative function interface member functions

A generative function is an object that is callable for some input type, and then, when called with any value of its input type, returns a distribution that produces traces of a given fixed type.

### Parameter store and gradient accumulator member functions

Parameter stores contain the state \f$\theta\f$ of some learnable parameters and the state of their gradient.
provide A parameter store type `ParameterStore` must provide `typename ParameterStore::accumulator_t`, and the accumulator type must have a constructor that takes a `ParameterStore` object as its only argument.
The gradient values are initialized to zero, and are accumulated by calling the `update_gradient()` member function on a gradient accumulator object.
The indirection between parameter stores and gradient accumulators is meant to facilitate multi-threaded gradient-based algorithms inwhich different threads accumulate gradients in parallel in thread-local gradient accumulators.

## Functionality

Distribution, trace, and generative function types that implement the above interface (or certain subsets of the interface) can be used with the following GenTL features:

### Modeling

GenTL contains some generic modeling constructs that can be used to compose more complex models from user-defined generative functions.

#### Sequential and Temporal Models

- Distribution type gentl::modeling::sequence::Model (with associated trace type gentl::modeling::sequence::Trace), that takes a user-defined generative function defining a transition kernel and gives a generative function that unrolls the kernel over a variable number of time steps.

### Inference and Learning

GenTL also contains building blocks for high-level implementation of inference and learning algorithms using traces.

#### Markov chain Monte Carlo (MCMC)

- gentl::mcmc::mh

- gentl::mcmc::hmc

- gentl::mcmc::mala

#### Sequential Monte Carlo (SMC)

- gentl::smc::ParticleSystem

#### Sampling importance resampling (SIR)

- gentl::sir::rolling_importance_resampling

#### Maximum likelihood

- gentl::sgd::train_supervised

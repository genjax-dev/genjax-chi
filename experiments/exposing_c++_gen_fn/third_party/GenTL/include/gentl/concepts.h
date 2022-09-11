/* Copyright 2021-2022 Massachusetts Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef GENTL_CONCEPTS_H
#define GENTL_CONCEPTS_H

#ifdef __cpp_concepts

#include <gentl/types.h>
#include <random>
#include <memory>

namespace gentl {

// TODO add backward_constraints
// TODO the distinction for bound and unbound genreative function may need some refinement

template <typename ChoiceBuffer>
concept RealArithmetic = requires(ChoiceBuffer a, ChoiceBuffer b, double c) {
    { a + b } -> std::convertible_to<ChoiceBuffer>;
    { c * a } -> std::convertible_to<ChoiceBuffer>;
    { b = a } -> std::convertible_to<ChoiceBuffer>;
    { b += a } -> std::convertible_to<ChoiceBuffer>;
};

// trace concepts

template <typename Trace>
concept HasScore = requires(Trace trace) {
    { trace.score() } -> std::same_as<double>;
};

template <typename Trace>
concept HasReturnValue = requires(Trace trace) {
    { trace.retufrn_value() };
};

template <typename Trace, typename Value>
concept ReturnTypeConvertibleTo = requires(Trace trace) {
    { trace.return_value() } -> std::convertible_to<Value>;
};

template <typename Trace, typename InputChange, typename Constraint>
concept Updatable = requires(Trace trace, InputChange input_change, Constraint constraint, std::mt19937 rng) {
    { trace.update(rng, input_change, constraint, UpdateOptions()) } -> std::same_as<double>;
};

template <typename Trace>
concept Revertible = requires(Trace trace) {
    { trace.revert() };
};

template <typename Trace>
concept Forkable = requires(Trace trace) {
    { trace.fork() } -> std::same_as<std::unique_ptr<Trace>>;
};


template <typename Trace, typename GradientAccumulator>
concept HasNonCompositionalParameterGradient = requires(Trace trace, GradientAccumulator accum) {
    { trace.parameter_gradient(accum) };
    { trace.parameter_gradient(accum, 1.0) };
};

template <typename Trace, typename GradientAccumulator, typename InputGradient, typename OutputGradient>
concept HasCompositionalParameterGradient = HasReturnValue<Trace> &&
        requires(Trace trace, GradientAccumulator accum, OutputGradient output) {
            { trace.parameter_gradient(accum, output) } -> std::same_as<InputGradient>;
            { trace.parameter_gradient(accum, output, 1.0) } -> std::same_as<InputGradient>;
};

template <typename Trace, typename Selection, typename ChoiceGradient>
concept HasNonCompositionalChoiceGradient = requires(Trace trace, Selection selection) {
    { trace.choice_gradient(selection) } -> std::same_as<const ChoiceGradient&>;
};

template <typename Trace, typename Selection, typename ChoiceGradient, typename InputGradient, typename OutputGradient>
concept HasCompositionalChoiceGradient = HasReturnValue<Trace> &&
        requires(Trace trace, Selection selection, OutputGradient output) {
            { trace.choice_gradient(selection, output) } -> std::same_as<std::pair<InputGradient, const ChoiceGradient&>>;
};

template <typename Trace, typename Selection, typename ChoiceBuffer>
concept TopLevelGradientBasedChoiceUpdates =
        HasNonCompositionalChoiceGradient<Trace,Selection,ChoiceBuffer> &&
        Updatable<Trace,gentl::change::NoChange,ChoiceBuffer> &&
        requires(Trace trace, Selection selection, ChoiceBuffer choices) {
            { trace.choices(selection) } -> std::convertible_to<ChoiceBuffer>;
        } &&
        RealArithmetic<ChoiceBuffer>;

// generative function concepts

template <typename GenFn, typename Input, typename Result>
concept Invocable = requires(GenFn fn, Input input) {
    { fn(input) } -> std::convertible_to<Result>;
};

template <typename Distribution, typename ParameterStore, typename Trace>
concept SimulatableDistribution =
        requires(Distribution dist, std::mt19937 rng, ParameterStore store) {
            { dist.simulate(rng, store, SimulateOptions()) } -> std::same_as<std::unique_ptr<Trace>>;
};

template <typename Distribution, typename ParameterStore, typename Constraint, typename Trace>
concept GeneratableDistribution =
        requires(Distribution dist, std::mt19937 rng, ParameterStore store, Constraint constraint) {
            { dist.generate(rng, store, constraint, GenerateOptions()) } -> std::same_as<std::pair<std::unique_ptr<Trace>,double>>;
};

template <typename Distribution, typename ParameterStore, typename Constraint, typename Trace>
concept InPlaceGeneratableDistribution =
requires(Distribution dist, std::mt19937 rng, ParameterStore store, Constraint constraint, Trace trace) {
    { dist.generate(trace, rng, store, constraint, GenerateOptions()) } -> std::same_as<double>;
};

template <typename GenFn, typename Input, typename ParameterStore, typename Trace>
concept SimulatableGenerativeFunction =
      requires(GenFn fn, Input input, std::mt19937 rng, ParameterStore store) {
          { fn(input).simulate(rng, store, SimulateOptions()) } -> std::same_as<std::unique_ptr<Trace>>;
      };

template <typename GenFn, typename Input, typename ParameterStore, typename Trace, typename Constraint>
concept GeneratableGenerativeFunction =
      requires(GenFn fn, Input input, std::mt19937 rng, ParameterStore store, Constraint constraint) {
          { fn(input).generate(rng, store, constraint, GenerateOptions()) } -> std::same_as<std::pair<std::unique_ptr<Trace>,double>>;
      };

template <typename GenFn, typename Input, typename ParameterStore, typename Constraint> 
concept AssessableGenerativeFunction =
      requires(GenFn fn, Input input, std::mt19937 rng, ParameterStore store, Constraint constraint) {
          { fn(input).assess(rng, store, constraint) } -> std::same_as<std::pair<typename GenFn::return_type, double>>;
      };


// TODO replace this with update variant:
template <typename GenFn, typename Input, typename ParameterStore, typename Trace, typename Constraint>
concept InPlaceGeneratableGenerativeFunction =
requires(GenFn fn, Input input, std::mt19937 rng, ParameterStore store, Constraint constraint, Trace trace) {
    { fn(input).generate(trace, rng, store, constraint, GenerateOptions()) } -> std::same_as<double>;
};

template <typename GenFn, typename Input, typename ParameterStore, typename Trace>
concept InPlaceSimulatableGenerativeFunction =
requires(GenFn fn, Input input, std::mt19937 rng, ParameterStore store, Trace trace) {
    { fn(input).simulate(rng, store, SimulateOptions(), trace) };
};


// parameter concepts

template <typename Accum>
concept GradientAccumulator = requires(Accum accum) {
    { accum.update_gradient() };
};

}

#endif

#endif //GENTL_CONCEPTS_H

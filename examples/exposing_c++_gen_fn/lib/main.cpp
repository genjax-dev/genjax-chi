// Copyright 2022 MIT Probabilistic Computing Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gentl/types.h>
#include <gentl/util/randutils.h>

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pybind11_helpers.h"

using gentl::GenerateOptions;
using gentl::SimulateOptions;
using gentl::UpdateOptions;

typedef float mean_t;
typedef float cov_t;

class LatentsSelection {};

class EmptyChoiceBuffer {};

typedef float latent_choices_t;

class RetvalChange {};

class GradientAccumulator {};

class ModelTrace;

class ModelParameters {};

class Model {
  typedef int return_type;
  friend class ModelTrace;

 private:
  mean_t mean_;
  cov_t cov_;

 public:
  template <typename RNGType>
  void exact_sample(latent_choices_t &latents, RNGType &rng) const {
    static std::normal_distribution<float> standard_normal_dist(0.0, 1.0);
    latents = standard_normal_dist(rng);
    return;
  }

  [[nodiscard]] float logpdf(const latent_choices_t &latents) const {
    static float logSqrt2Pi = 0.5 * std::log(2 * M_PI);
    auto w = -0.5 * (latents * latents) - logSqrt2Pi;
    return w;
  }

  template <typename RNGType>
  [[nodiscard]] std::pair<float, float> importance_sample(
      latent_choices_t &latents, RNGType &rng) const {
    exact_sample(latents, rng);
    float log_weight = 0.0;
    return {logpdf(latents), log_weight};
  }

 public:
  Model(mean_t mean, cov_t cov)
      : mean_{std::move(mean)}, cov_{std::move(cov)} {}

  // simulate into a new trace object
  template <typename RNGType>
  std::unique_ptr<ModelTrace> simulate(RNGType &rng,
                                       ModelParameters &parameters,
                                       const SimulateOptions &) const;

  // simulate into an existing trace object (overwriting existing contents)
  template <typename RNGType>
  void simulate(RNGType &rng, ModelParameters &parameters,
                const SimulateOptions &, ModelTrace &trace) const;

  // generate into a new trace object
  template <typename RNGType>
  std::pair<std::unique_ptr<ModelTrace>, float> generate(
      const EmptyChoiceBuffer &constraints, RNGType &rng,
      ModelParameters &parameters, const GenerateOptions &) const;

  // generate into an existing trace object (overwriting existing contents)
  template <typename RNGType>
  float generate(ModelTrace &trace, const EmptyChoiceBuffer &constraints,
                 RNGType &rng, ModelParameters &parameters,
                 const GenerateOptions &) const;

  // equivalent to generate but without returning a trace
  template <typename RNG>
  std::pair<int, float> assess(RNG &, ModelParameters &,
                               const latent_choices_t &constraints) const;

  template <typename RNG>
  std::pair<int, float> assess(RNG &, ModelParameters &,
                               const EmptyChoiceBuffer &constraints) const;
};

class ModelTrace {
  friend class Model;

 private:
  Model model_;
  float score_;
  latent_choices_t latents_;
  latent_choices_t alternate_latents_;
  latent_choices_t latent_gradient_;
  bool can_be_reverted_;

 private:
  ModelTrace(Model model, float score, latent_choices_t &&latents)
      : model_{std::move(model)},
        score_{score},
        latents_{latents},
        can_be_reverted_{false} {}

 public:
  ModelTrace() = delete;
  ModelTrace(const ModelTrace &other) = delete;
  ModelTrace(ModelTrace &&other) = delete;
  ModelTrace &operator=(const ModelTrace &other) = delete;
  ModelTrace &operator=(ModelTrace &&other) noexcept = delete;

  [[nodiscard]] float score() const;
  [[nodiscard]] const latent_choices_t &choices() const;
  [[nodiscard]] const latent_choices_t &choices(
      const LatentsSelection &selection) const;
  const latent_choices_t &choice_gradient(const LatentsSelection &selection);

  template <typename RNG>
  float update(RNG &, const gentl::change::NoChange &,
               const latent_choices_t &constraints,
               const UpdateOptions &options);
  const latent_choices_t &backward_constraints();

  void revert();
};

// ****************************
// *** Model implementation ***
// ****************************

template <typename RNGType>
std::unique_ptr<ModelTrace> Model::simulate(
    RNGType &rng, ModelParameters &parameters,
    const SimulateOptions &options) const {
  latent_choices_t latents;
  exact_sample(latents, rng);
  auto log_density = logpdf(latents);
  // note: this copies the model
  return std::unique_ptr<ModelTrace>(
      new ModelTrace(*this, log_density, std::move(latents)));
}

template <typename RNGType>
void Model::simulate(RNGType &rng, ModelParameters &parameters,
                     const SimulateOptions &options, ModelTrace &trace) const {
  exact_sample(trace.latents_, rng);
  trace.score_ = logpdf(trace.latents_);
  trace.can_be_reverted_ = false;
}

template <typename RNGType>
std::pair<std::unique_ptr<ModelTrace>, float> Model::generate(
    const EmptyChoiceBuffer &constraints, RNGType &rng,
    ModelParameters &parameters, const GenerateOptions &options) const {
  latent_choices_t latents;
  auto [log_density, log_weight] = importance_sample(latents, rng);
  std::unique_ptr<ModelTrace> trace = nullptr;
  trace = std::unique_ptr<ModelTrace>(
      new ModelTrace(*this, log_density, std::move(latents)));
  return {std::move(trace), log_weight};
}

template <typename RNGType>
float Model::generate(ModelTrace &trace, const EmptyChoiceBuffer &constraints,
                      RNGType &rng, ModelParameters &parameters,
                      const GenerateOptions &options) const {
  trace.model_ = *this;
  auto [log_density, log_weight] = importance_sample(trace.latents_, rng);
  trace.score_ = log_density;
  float score = logpdf(trace.latents_);
  trace.can_be_reverted_ = false;
  return log_weight;
}

template <typename RNG>
std::pair<int, float> Model::assess(RNG &, ModelParameters &parameters,
                                    const latent_choices_t &constraints) const {
  return {-1, logpdf(constraints)};
}

template <typename RNG>
std::pair<int, float> Model::assess(
    RNG &, ModelParameters &parameters,
    const EmptyChoiceBuffer &constraints) const {
  return {-1, 0.0};
}

// ****************************
// *** Trace implementation ***
// ****************************

float ModelTrace::score() const { return score_; }

const latent_choices_t &ModelTrace::choices(
    const LatentsSelection &selection) const {
  return latents_;
}

const latent_choices_t &ModelTrace::choices() const { return latents_; }

void ModelTrace::revert() {
  if (!can_be_reverted_)
    throw std::logic_error(
        "log_weight is only available between calls to update and revert");
  can_be_reverted_ = false;
  std::swap(latents_, alternate_latents_);
}

const latent_choices_t &ModelTrace::backward_constraints() {
  return alternate_latents_;
}

template <typename RNG>
float ModelTrace::update(RNG &, const gentl::change::NoChange &,
                         const latent_choices_t &latents,
                         const UpdateOptions &options) {
  if (options.save()) {
    std::swap(latents_, alternate_latents_);
    latents_ = latents;  // copy assignment
    can_be_reverted_ = true;
  } else {
    latents_ = latents;  // copy assignment
                         // can_be_reverted_ keeps its previous value
  };
  float new_log_density = model_.logpdf(latents_);
  float log_weight = new_log_density - score_;
  score_ = new_log_density;
  return log_weight;
}

// Here, we expose C-like functions to pybind11.
namespace extern_gen_fn {

template <typename T>
void cpu_simulate(void *out, const void **in) {
  const T *mean = reinterpret_cast<const T *>(in[0]);
  const T *cov = reinterpret_cast<const T *>(in[1]);
  unsigned int seed = 314159;
  gentl::randutils::seed_seq_fe128 seed_seq{seed};
  std::mt19937 rng(seed_seq);
  ModelParameters unused{};
  Model model{*mean, *cov};
  auto trace = model.simulate(rng, unused, SimulateOptions());
  T *result = reinterpret_cast<T *>(out);
  *result = trace->choices();
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_simulate_f32"] = EncapsulateFunction(cpu_simulate<float>);
  return dict;
}

PYBIND11_MODULE(extern_gen_fn, m) { m.def("registrations", &Registrations); }

}  // namespace extern_gen_fn

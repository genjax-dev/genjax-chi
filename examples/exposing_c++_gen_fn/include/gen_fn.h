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

#ifndef _gen_fn_gen_fn_H_
#define _gen_fn_gen_fn_H_

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

using gentl::GenerateOptions;
using gentl::SimulateOptions;
using gentl::UpdateOptions;

namespace gen_fn {

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

}  // namespace gen_fn

#endif

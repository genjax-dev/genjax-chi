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

#include "gen_fn.h"

#include <gentl/types.h>
#include <gentl/util/randutils.h>

#include <array>
#include <chrono>
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

using namespace gen_fn;

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

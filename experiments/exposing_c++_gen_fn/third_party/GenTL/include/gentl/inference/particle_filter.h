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

#ifndef GENTL_PARTICLE_FILTER_H
#define GENTL_PARTICLE_FILTER_H

#include <cmath>
#include <random>
#include <algorithm>

#include <gentl/concepts.h>
#include <gentl/types.h>
#include <gentl/util/mathutils.h>

using gentl::GenerateOptions;
using gentl::UpdateOptions;

namespace gentl::smc {

template <typename Trace, typename RNGType>
#ifdef __cpp_concepts
requires Forkable<Trace>
#endif
class ParticleSystem {
private:
    size_t num_particles_;
    std::vector<std::unique_ptr<Trace>> traces_;
    std::vector<std::unique_ptr<Trace>> traces_tmp_; // initially contains all nullptrs
    std::vector<double> log_weights_;
    std::vector<double> log_normalized_weights_;
    std::vector<double> two_times_log_normalized_weights_;
    std::vector<double> normalized_weights_;
    std::vector<size_t> parents_;
    std::vector<Trace*> trace_nonowning_ptrs_;
    RNGType& rng_; // TODO this will need to be replaced with a seed_seq for the multi-threaded version.
    double log_ml_estimate_;

    double normalize_weights() {
        double log_total_weight = gentl::mathutils::logsumexp(log_weights_);
        for (size_t i = 0; i < num_particles_; i++) {
            log_normalized_weights_[i] = log_weights_[i] - log_total_weight;
            two_times_log_normalized_weights_[i] = 2.0 * log_normalized_weights_[i];
            normalized_weights_[i] = std::exp(log_normalized_weights_[i]);
        }
        return log_total_weight;
    }

    void multinomial_resampling() {
        std::discrete_distribution<size_t> dist(normalized_weights_.cbegin(), normalized_weights_.cend());
        for (size_t i = 0; i < num_particles_; i++) {
            parents_[i] = dist(rng_);
        }
    }

public:
    ParticleSystem(size_t num_particles, RNGType& rng) :
            num_particles_(num_particles),
            traces_(num_particles),
            traces_tmp_(num_particles),
            log_weights_(num_particles),
            log_normalized_weights_(num_particles),
            two_times_log_normalized_weights_(num_particles),
            normalized_weights_(num_particles),
            parents_(num_particles),
            trace_nonowning_ptrs_(num_particles),
            rng_(rng),
            log_ml_estimate_(0.0) {
    }

    template <typename Model, typename Parameters, typename Constraints>
    void init_step(const Model& model, Parameters& parameters, const Constraints& constraints) {
        // TODO make prepare_for_gradient optional for each generate() and update() call
        for (size_t i = 0; i < num_particles_; i++) {
            auto [trace_ptr, log_weight] = model.generate(rng_, parameters, constraints, GenerateOptions());
            traces_[i] = std::move(trace_ptr);
            trace_nonowning_ptrs_[i] = traces_[i].get();
            log_weights_[i] = log_weight;
        }
    }

    // TODO document requirements (and non-requirements) associated with normalization of the model
    // (the model doesn't need to be normalized, and the normalizing constant can change)
    // technically, this means you could just implement this via a ModelChange without constraints
    // but the distinction is still useful for the common case when the models are normalized
    template <typename InputChange, typename Constraints>
    #ifdef __cpp_concepts
        requires Updatable<Trace, InputChange, Constraints>
    #endif
    void step(const InputChange& input_change, const Constraints& constraints) {
        for (size_t i = 0; i < traces_.size(); i++)
            log_weights_[i] += traces_[i]->update(rng_, input_change, constraints, UpdateOptions());
    }

    [[nodiscard]] double effective_sample_size() const {
        return std::exp(-gentl::mathutils::logsumexp(two_times_log_normalized_weights_));
    }

    double resample() {
        // writes to normalized_weights_
        double log_total_weight = normalize_weights();
        log_ml_estimate_ += log_total_weight - std::log(num_particles_);

        // reads from normalized_weights_ and writes to parents_
        multinomial_resampling();

        // TODO do this in a way that keeps traces in the same thread if you can to minimize data movement
        // and avoids ruining the cache every time we resample
        for (size_t i = 0; i < num_particles_; i++)
            traces_tmp_[i] = traces_[parents_[i]]->fork();
        for (size_t i = 0; i < num_particles_; i++) {
            traces_[i] = std::move(traces_tmp_[i]); // move assignment
            trace_nonowning_ptrs_[i] = traces_[i].get();
        }
        std::fill(log_weights_.begin(), log_weights_.end(), 0.0);
        return log_total_weight;
    }

    [[nodiscard]] double log_marginal_likelihood_estimate() const {
        return log_ml_estimate_ + gentl::mathutils::logsumexp(log_weights_) - std::log(num_particles_);
    }

    // so that user can run rejuvenation moves on them
    const std::vector<Trace*>& traces() const {
        return trace_nonowning_ptrs_;
    }

};


}

#endif //GENTL_PARTICLE_FILTER_H

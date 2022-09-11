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

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <iostream>
#include <Eigen/Dense>

#include <gentl/types.h>
#include <gentl/util/randutils.h>
#include <gentl/inference/particle_filter.h>

using gentl::GenerateOptions;
using gentl::UpdateOptions;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::indexing::all;
using std::valarray;

// *******************************************************************
// ** forward algorithm to compute ground truth marginal likelihood **
// *******************************************************************

double hmm_forward_alg(const VectorXd& prior,
                       const MatrixXd& emission_dists,
                       const MatrixXd& transition_dists,
                       const std::vector<size_t>& emissions) {
    assert(prior.rows() == emission_dists.cols());
    assert(prior.rows() == transition_dists.cols());
    assert(transition_dists.rows() == transition_dists.rows());
    double marginal_likelihood = 1.0;
    VectorXd alpha = prior; // copy
    for (auto emission : emissions) {
        auto likelihoods = emission_dists(emission, all).transpose();
        VectorXd prev_posterior = (alpha.array() * likelihoods.array()).matrix();
        double denom = prev_posterior.sum();
        prev_posterior /= denom;
        alpha = transition_dists * prev_posterior;
        marginal_likelihood *= denom;
    }
    return marginal_likelihood;
}

TEST_CASE("hmm forward algorithm", "[particle filtering]") {

    // test the hmm_forward_alg on a hand-calculated example
    VectorXd prior {{0.4, 0.6}};
    MatrixXd emission_dists {
            {0.1, 0.9}, // dist for state 0
            {0.7, 0.3} // dist for state 1
    };
    emission_dists.transposeInPlace();
    MatrixXd transition_dists {
            {0.5, 0.5}, // dist. for state 0
            {0.2, 0.8} // dist. for state 1
    };
    transition_dists.transposeInPlace();
    std::vector<size_t> obs {1, 0};
    double expected_marginal_likelihood = 0.0;
    // z = [0, 0]
    expected_marginal_likelihood += prior(0) * transition_dists(0, 0) * emission_dists(obs[0], 0) * emission_dists(obs[1], 0);
    // z = [0, 1]
    expected_marginal_likelihood += prior(0) * transition_dists(1, 0) * emission_dists(obs[0], 0) * emission_dists(obs[1], 1);
    // z = [1, 0]
    expected_marginal_likelihood += prior(1) * transition_dists(0, 1) * emission_dists(obs[0], 1) * emission_dists(obs[1], 0);
    // z = [1, 1]
    expected_marginal_likelihood += prior(1) * transition_dists(1, 1)* emission_dists(obs[0], 1) * emission_dists(obs[1], 1);
    auto actual_marginal_likelihood = hmm_forward_alg(prior, emission_dists, transition_dists, obs);
    REQUIRE(std::abs(actual_marginal_likelihood - expected_marginal_likelihood) < 1e-16);
}

// ************************************************************************
// ** minimum HMM implementation for testing particle filter correctness **
// ************************************************************************

class HMMParams {
    friend class HMM;
    friend class HMMTrace;
private:
    const size_t num_states_;
    const MatrixXd emission_matrix_;
    std::discrete_distribution<size_t> prior_dist_;
    std::vector<std::discrete_distribution<size_t>> transition_dists_;
public:
    HMMParams(const VectorXd& prior, const MatrixXd& emission_matrix, const MatrixXd& transition_matrix) :
            emission_matrix_(emission_matrix), num_states_(emission_matrix.cols()) {
        if (transition_matrix.rows() != num_states_ || transition_matrix.cols() != num_states_)
            throw std::logic_error("dimension mismatch");
        prior_dist_ = std::discrete_distribution<size_t>(prior.cbegin(), prior.cend());
        for (auto i = 0; i < num_states_; i++) {
            auto column_vector = transition_matrix(all, i);
            assert(std::abs(column_vector.sum() - 1.0) < 1e-16);
            transition_dists_.emplace_back(column_vector.begin(), column_vector.end());
        }
    }
};

class ParameterStore {};

class Extend {};

struct NewObservation {
    size_t value;
    NewObservation(size_t value_) : value(value_) {}
};

class HMMTrace;

class HMM {
    friend class HMMTrace;
    size_t num_time_steps_;
    HMMParams& params_;
private:
public:
    explicit HMM(HMMParams& params) : num_time_steps_(0), params_(params) {}
    std::pair<std::unique_ptr<HMMTrace>,double> generate(
            std::mt19937& rng, ParameterStore&, const NewObservation& observation,
            const GenerateOptions& options) const;

};

class HMMTrace {
    friend class HMM;
    HMM model_;
    std::vector<size_t> emissions_;
    std::vector<size_t> latents_;
private:
    HMMTrace(const HMMTrace& other) = default;
    HMMTrace(HMM model, size_t emission, size_t latent) :
            emissions_(std::initializer_list<size_t>({emission})),
            latents_(std::initializer_list<size_t>({latent})),
            model_(model) {}
public:
    double update(
            std::mt19937& rng, const Extend&, const NewObservation& observation,
            const UpdateOptions& options) {
        if (options.save() || options.precompute_gradient())
            throw std::logic_error("not implemented");
        size_t latent = model_.params_.transition_dists_[latents_.back()](rng);
        double log_weight = std::log(model_.params_.emission_matrix_(observation.value, latent));
        latents_.emplace_back(latent);
        emissions_.emplace_back(observation.value);
        model_.num_time_steps_ += 1;
        return log_weight;
    }
    std::unique_ptr<HMMTrace> fork() {
        // NOTE: this trace implementation is not efficient, it copies entire histories unnecessarily
        return std::unique_ptr<HMMTrace>(new HMMTrace(*this));
    }
};

std::pair<std::unique_ptr<HMMTrace>,double> HMM::generate(
        std::mt19937& rng, ParameterStore&, const NewObservation& observation,
        const GenerateOptions& options) const {
    if (options.precompute_gradient())
        throw std::logic_error("not implemented");
    size_t latent = params_.prior_dist_(rng);
    double log_weight = std::log(params_.emission_matrix_(observation.value, latent));
    auto trace = std::unique_ptr<HMMTrace>(new HMMTrace(*this, observation.value, latent));
    return {std::move(trace), log_weight};
}


TEST_CASE("hmm particle filter", "[particle filtering]") {

    gentl::randutils::seed_seq_fe128 seed_seq {0};
    std::mt19937 rng(seed_seq);

    VectorXd prior {{0.2, 0.3, 0.5}};
    MatrixXd emission_matrix {
            {0.1, 0.2, 0.7},
            {0.2, 0.7, 0.1},
            {0.7, 0.2, 0.1}
    };
    emission_matrix.transposeInPlace();
    MatrixXd transition_matrix {
            {0.4, 0.4, 0.2},
            {0.2, 0.3, 0.5},
            {0.9, 0.05, 0.05}
    };
    transition_matrix.transposeInPlace();
    HMMParams params {
        prior, emission_matrix, transition_matrix
    };

    HMM model {params};

    // test particle filter
    std::vector<size_t> data = {0, 0, 1, 2};
    double expected = std::log(hmm_forward_alg(prior, emission_matrix, transition_matrix, data));

    std::vector<NewObservation> observations;
    for (auto datum : data)
        observations.emplace_back(datum);

    size_t num_particles = 10000;
    gentl::smc::ParticleSystem<HMMTrace,std::mt19937> filter{num_particles, rng};
    auto observations_it = observations.cbegin();
    ParameterStore store{};
    filter.init_step(model, store, *observations_it++);
    using std::cerr, std::endl;
    while (observations_it != observations.cend()) {
        filter.step(Extend{}, *observations_it++);
        double ess = filter.effective_sample_size();
        cerr << "effective sample size: " << ess << endl;
        double log_weight = filter.resample();
        cerr << "log weight from resample: " << log_weight << endl;
    }
    double actual = filter.log_marginal_likelihood_estimate();
    cerr << "actual: " << actual << ", expected: " << expected << endl;
    REQUIRE(std::abs(actual - expected) < 0.02);
}

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

#include <catch2/catch.hpp>

#include <gentl/util/randutils.h>
#include <gentl/modeling/sequence.h>
#include <gentl/inference/particle_filter.h>
#include <iostream>
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::indexing::all;

namespace sequence = gentl::modeling::sequence;

struct Parameters {
    VectorXd prior;
    MatrixXd emission_matrix;
    MatrixXd transition_matrix;
    Parameters(VectorXd prior_, MatrixXd emission_matrix_, MatrixXd transition_matrix_) :
        prior(prior_), emission_matrix(emission_matrix_), transition_matrix(transition_matrix_) {}
};

class KernelTrace;
class BoundKernel {
    long input_;
public:
    explicit BoundKernel(long input) : input_(input) { }
    template <typename RNGType>
    std::pair<std::unique_ptr<KernelTrace>,double> generate(
            RNGType& rng, Parameters& parameters, const long& observation,
            const GenerateOptions& options) const;
    template <typename RNGType>
    std::pair<std::unique_ptr<KernelTrace>,double> generate(
            RNGType& rng, Parameters& parameters, const sequence::constraints::Empty&,
            const GenerateOptions& options) const;
};
struct Kernel {
    BoundKernel operator()(long input) const { return BoundKernel(input); }
};
constexpr Kernel kernel;

struct KernelTrace {
    long hidden_state_;
    long observation_;
    KernelTrace(long hidden_state, long observation) : hidden_state_(hidden_state), observation_(observation) {}
    [[nodiscard]] long return_value() const {
        return hidden_state_;
    }
};

template <typename RNGType>
std::pair<std::unique_ptr<KernelTrace>,double> BoundKernel::generate(
        RNGType& rng, Parameters& parameters, const long& observation, const GenerateOptions& options) const {
    assert(!options.precompute_gradient());
    std::discrete_distribution<long> dynamics;
    if (input_ == -1) {
        // initial state
        dynamics = std::discrete_distribution<long>(parameters.prior.cbegin(), parameters.prior.cend());
    } else {
        auto column_vector = parameters.transition_matrix(all, input_);
        assert(std::abs(column_vector.sum() - 1.0) < 1e-16);
        dynamics = std::discrete_distribution<long>(column_vector.cbegin(), column_vector.cend());
    }
    long hidden_state = dynamics(rng);
    double log_weight = std::log(parameters.emission_matrix(observation, hidden_state));
    auto trace = std::make_unique<KernelTrace>(hidden_state, observation);
    return {std::move(trace), log_weight};
}


TEST_CASE("sequence hmm", "[sequence, particle filtering]") {
    using ValueType = long;
    using Model = sequence::Model<Kernel,KernelTrace,ValueType,Parameters>;
    using Trace = sequence::Trace<Kernel,KernelTrace,ValueType,Parameters>;
    using std::cerr, std::endl;

    // random seed
    gentl::randutils::seed_seq_fe128 seed_seq {0};
    std::mt19937 rng(seed_seq);

    // define parameters
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
    Parameters parameters(prior, emission_matrix, transition_matrix);

    // initial distribution
    ValueType init_value = -1;
    Model model(kernel, init_value);

    // test particle filter
    size_t num_particles = 10000;
    gentl::smc::ParticleSystem<Trace,std::mt19937> filter{num_particles, rng};
    std::vector<long> obs = {0, 0, 1, 2};
    filter.init_step(model, parameters, sequence::constraints::empty);
    auto it = obs.begin();
    while (it != obs.end()) {
        filter.step(sequence::modelchange::extend, sequence::constraints::NewStepOnly<long>(*it++));
        filter.resample();
    }
    double estimate = filter.log_marginal_likelihood_estimate();
    cerr << "actual: " << estimate;
//    << ", expected: " << expected << endl;
//    REQUIRE(std::abs(actual - expected) < 0.02);

}

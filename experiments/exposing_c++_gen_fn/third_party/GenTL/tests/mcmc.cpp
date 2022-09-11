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

#include <utility>
#include <vector>
#include <stdexcept>
#include <memory>
#include <array>
#include <iostream>

#include <Eigen/Dense>

#include <gentl/util/randutils.h>
#include <gentl/inference/mcmc.h>
#include <gentl/types.h>

template<typename Iter, typename Val>
#ifdef __cpp_concepts
    requires std::input_iterator<Iter>
#endif
Iter my_find(Iter b, Iter e, Val v)
{
    // ...
    return b;
}

void doit() {
    std::vector<int> vec = {1, 2, 3};
    auto iter = my_find(vec.begin(), vec.end(), 2);
    std::cout << *iter << std::endl;
}

using gentl::SimulateOptions;
using gentl::UpdateOptions;
using gentl::GenerateOptions;

// TODO support parameters
// TODO support subset of addresses

// ****************************
// *** Model implementation ***
// ****************************

constexpr size_t latent_dimension = 2;
typedef Eigen::Vector<double,latent_dimension> mean_t;
typedef Eigen::Matrix<double,latent_dimension,latent_dimension> cov_t;

// Selection types

class LatentsSelection {};

// Choice buffer types

class EmptyChoiceBuffer {};

typedef Eigen::Array<double,latent_dimension,1> latent_choices_t;

// learnable parameters (there are none)

class Parameters {};

// Trace and Model

class Trace;

class Model {
    friend class Trace;

private:
    typedef Eigen::LLT<Eigen::Matrix<double,latent_dimension,latent_dimension>> Chol;
    mean_t mean_;
    cov_t cov_;
    cov_t precision_;
    Chol chol_;

public:
    template <typename RNGType>
    void exact_sample(latent_choices_t& latents, RNGType& rng) const {
        static std::normal_distribution<double> standard_normal_dist(0.0, 1.0);
        for (auto& x : latents)
            x = standard_normal_dist(rng);
        latents = (mean_ + (chol_.matrixL() * latents.matrix())).array();
    }

    [[nodiscard]] double logpdf(const latent_choices_t& latents) const {
        static double logSqrt2Pi = 0.5*std::log(2*M_PI);
        double quadform = chol_.matrixL().solve(latents.matrix() - mean_).squaredNorm();
        return std::exp(-static_cast<double>(latent_dimension)*logSqrt2Pi - 0.5*quadform) / chol_.matrixL().determinant();
    }

    template <typename RNGType>
    [[nodiscard]] std::pair<double,double> importance_sample(latent_choices_t& latents, RNGType& rng) const {
        exact_sample(latents, rng);
        double log_weight = 0.0;
        return {logpdf(latents), log_weight};
    }

    void logpdf_grad(latent_choices_t& latent_gradient, const latent_choices_t& latents) const {
        // gradient wrt value x is -x
        latent_gradient = (-precision_ * (latents.matrix() - mean_)).array();
    }


public:
    Model(mean_t mean, cov_t cov) :
            mean_{std::move(mean)}, cov_{std::move(cov)}, chol_{cov}, precision_{cov.inverse()} {
        if (chol_.info() != Eigen::Success)
            throw std::logic_error("decomposition failed!");
    }

    // simulate into a new trace object
    template <typename RNGType>
    std::unique_ptr<Trace> simulate(RNGType& rng, Parameters& parameters, const SimulateOptions& options) const;

    // simulate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    void simulate(RNGType& rng, Parameters& parameters, const SimulateOptions& options, Trace& trace) const;

    // generate into a new trace object
    template <typename RNGType>
    std::pair<std::unique_ptr<Trace>,double> generate(RNGType& rng, Parameters& parameters,
                                                      const EmptyChoiceBuffer& constraints,
                                                      const GenerateOptions& options) const;

    // generate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    double generate(RNGType& rng, Parameters& parameters, const EmptyChoiceBuffer& constraints,
                    const GenerateOptions& options, Trace& trace) const;

    // equivalent to generate but without returning a trace
    template <typename RNGType>
    std::pair<int, double> assess(RNGType& rng, Parameters& parameters, const latent_choices_t& constraints) const;
};

class Trace {
    friend class Model;
    typedef int return_type;
private:
    Model model_;
    double score_;
    latent_choices_t latents_;
    latent_choices_t alternate_latents_;
    latent_choices_t latent_gradient_;
    bool can_be_reverted_;
    bool gradients_computed_;
private:
    // initialize trace without precomputed gradient
    Trace(Model model, double score, latent_choices_t&& latents) :
        model_{std::move(model)}, score_{score}, latents_{latents},
        can_be_reverted_{false}, gradients_computed_{false} {}
    // initialize trace with gradient precomputed
    Trace(Model model, double score, latent_choices_t&& latents, latent_choices_t&& latent_gradient) :
    model_{std::move(model)}, score_{score}, latents_{latents}, latent_gradient_{latent_gradient},
        can_be_reverted_{false}, gradients_computed_{true} {}
public:
    Trace() = delete;
    Trace(const Trace& other) = delete;
    Trace(Trace&& other) = delete;
    Trace& operator=(const Trace& other) = delete;
    Trace& operator=(Trace&& other) noexcept = delete;

    [[nodiscard]] double score() const;
    [[nodiscard]] const latent_choices_t& choices() const;
    [[nodiscard]] const latent_choices_t& choices(const gentl::selection::All&) const;
    [[nodiscard]] const latent_choices_t& choices(const LatentsSelection& selection) const;
    const latent_choices_t& choice_gradient(const LatentsSelection& selection);
    template <typename RNG>
    double update(RNG&, const gentl::change::NoChange&, const latent_choices_t& constraints, const UpdateOptions& options);
    const latent_choices_t& backward_constraints();
    void revert();
};


// ****************************
// *** Model implementation ***
// ****************************

template <typename RNGType>
std::unique_ptr<Trace> Model::simulate(RNGType& rng, Parameters& parameters, const SimulateOptions& options) const {
    latent_choices_t latents;
    exact_sample(latents, rng);
    auto log_density = logpdf(latents);
    if (options.precompute_gradient()) {
        latent_choices_t latent_gradient;
        logpdf_grad(latent_gradient, latents);
        // note: this copies the model
        return std::unique_ptr<Trace>(new Trace(*this, log_density, std::move(latents), std::move(latent_gradient)));
    } else {
        // note: this copies the model
        return std::unique_ptr<Trace>(new Trace(*this, log_density, std::move(latents)));
    }
}

template <typename RNGType>
void Model::simulate(RNGType& rng, Parameters& parameters, const SimulateOptions& options, Trace& trace) const {
    exact_sample(trace.latents_, rng);
    trace.score_ = logpdf(trace.latents_);
    if ((trace.gradients_computed_ = options.precompute_gradient()))
        logpdf_grad(trace.latent_gradient_, trace.latents_);
    trace.can_be_reverted_ = false;
}

template <typename RNGType>
std::pair<std::unique_ptr<Trace>,double> Model::generate(
        RNGType& rng, Parameters& parameters, const EmptyChoiceBuffer& constraints,
        const GenerateOptions& options) const {
    latent_choices_t latents;
    auto [log_density, log_weight] = importance_sample(latents, rng);
    std::unique_ptr<Trace> trace = nullptr;
    if (options.precompute_gradient()) {
        latent_choices_t latent_gradient;
        logpdf_grad(latent_gradient, latents);
        trace = std::unique_ptr<Trace>(new Trace(*this, log_density, std::move(latents), std::move(latent_gradient)));
    } else {
        trace = std::unique_ptr<Trace>(new Trace(*this, log_density, std::move(latents)));
    }
    return {std::move(trace), log_weight};
}

template <typename RNGType>
double Model::generate(
        RNGType& rng, Parameters& parameters, const EmptyChoiceBuffer& constraints,
        const GenerateOptions& options, Trace& trace) const {
    trace.model_ = *this;
    auto [log_density, log_weight] = importance_sample(trace.latents_, rng);
    trace.score_ = log_density;
    double score = logpdf(trace.latents_);
    if ((trace.gradients_computed_ = options.precompute_gradient()))
        logpdf_grad(trace.latent_gradient_, trace.latents_);
    trace.can_be_reverted_ = false;
    return log_weight;
}

template <typename RNGType>
std::pair<int, double> Model::assess(RNGType& rng, Parameters& parameters, const latent_choices_t& constraints) const {
    return {-1, logpdf(constraints)};
}

// ****************************
// *** Trace implementation ***
// ****************************

double Trace::score() const {
    return score_;
}

const latent_choices_t& Trace::choices(const LatentsSelection& selection) const {
    return latents_;
}

const latent_choices_t& Trace::choices(const gentl::selection::All&) const {
    return latents_;
}

const latent_choices_t& Trace::choices() const {
    return latents_;
}

const latent_choices_t& Trace::backward_constraints() {
    return alternate_latents_;
}

void Trace::revert() {
    if (!can_be_reverted_)
        throw std::logic_error("log_weight is only available between calls to update and revert");
    can_be_reverted_ = false;
    std::swap(latents_, alternate_latents_);
    gradients_computed_ = false;
}

const latent_choices_t& Trace::choice_gradient(const LatentsSelection& selection) {
    if (!gradients_computed_)
        model_.logpdf_grad(latent_gradient_, latents_);
    gradients_computed_ = true;
    return  latent_gradient_;
}

template <typename RNGType>
double Trace::update(RNGType&, const gentl::change::NoChange&, const latent_choices_t& constraints,
                     const UpdateOptions& options) {
    if (options.save()) {
        std::swap(latents_, alternate_latents_);
        can_be_reverted_ = true;
    } else {
        // can_be_reverted_ keeps its previous value
    };
    latents_ = constraints; // copy assignment
    double new_log_density = model_.logpdf(latents_);
    double log_weight = new_log_density - score_;
    score_ = new_log_density;
    if ((gradients_computed_ = options.precompute_gradient()))
        model_.logpdf_grad(latent_gradient_, latents_);
    return log_weight;
}

TEST_CASE("smoke test", "[mcmc]") {
    doit();

    size_t hmc_cycles_per_iter = 2;
    size_t mala_cycles_per_iter = 2;
    size_t mh_cycles_per_iter = 2;
    size_t hmc_leapfrog_steps = 2;
    double hmc_eps = 0.01;
    double mala_tau = 0.01;
    size_t num_iters = 1000;
    uint32_t seed = 0;

    // initialize RNG
    gentl::randutils::seed_seq_fe128 seed_seq {seed};
    std::mt19937 rng(seed_seq);

    // define the model and proposal
    mean_t mean {0.0, 0.0};

    cov_t target_covariance {{1.0, 0.95},
                             {0.95, 1.0}};
    Model model {mean, target_covariance};

    cov_t proposal_covariance {{1.0, 0.95},
                               {0.95, 1.0}};

    auto proposal = [mean, proposal_covariance](const Trace& trace) {
        static Model proposal_distribution {mean, proposal_covariance};
        return proposal_distribution;
    };

    // generate initial trace and choice buffers
    Parameters parameters {};
    auto [trace, log_weight] = model.generate(rng, parameters, EmptyChoiceBuffer{},
                                              GenerateOptions().precompute_gradient(true));
    LatentsSelection hmc_selection;
    LatentsSelection mala_selection;
    auto proposal_trace = proposal(*trace).simulate(rng, parameters, SimulateOptions());

    latent_choices_t hmc_momenta_buffer {trace->choices(hmc_selection)}; // copy constructor
    latent_choices_t hmc_values_buffer {trace->choices(hmc_selection)}; // copy constructor
    latent_choices_t mala_buffer_1 {trace->choices(mala_selection)}; // copy constructor
    latent_choices_t mala_buffer_2 {trace->choices(mala_selection)}; // copy constructor

    // do some MALA and HMC on the latent variables (without allocating any memory inside the loop)
    std::vector<mean_t> history(num_iters);
    size_t hmc_num_accepted = 0;
    size_t mala_num_accepted = 0;
    size_t mh_num_accepted = 0;
    for (size_t iter = 0; iter < num_iters; iter++) {
        history[iter] = trace->choices(LatentsSelection{}).matrix();
        for (size_t cycle = 0; cycle < hmc_cycles_per_iter; cycle++) {
            hmc_num_accepted += gentl::mcmc::hmc(
                    *trace, hmc_selection, hmc_leapfrog_steps, hmc_eps,
                    hmc_momenta_buffer, hmc_values_buffer, rng);
        }
        for (size_t cycle = 0; cycle < mala_cycles_per_iter; cycle++) {
            mala_num_accepted += gentl::mcmc::mala(
                    *trace, mala_selection, mala_tau, mala_buffer_1, mala_buffer_2, rng);
        }
        for (size_t cycle = 0; cycle < mh_cycles_per_iter; cycle++) {
            mh_num_accepted += gentl::mcmc::mh(
                    rng, *trace, proposal, parameters, *proposal_trace, true);
        }
    }

    double hmc_accept_rate = static_cast<double>(hmc_num_accepted) / static_cast<double>(num_iters * hmc_cycles_per_iter);
    double mala_accept_rate = static_cast<double>(mala_num_accepted) / static_cast<double>(num_iters * mala_cycles_per_iter);
    double mh_accept_rate = static_cast<double>(mh_num_accepted) / static_cast<double>(num_iters * mh_cycles_per_iter);

    REQUIRE(hmc_accept_rate > 0.0);
    REQUIRE(hmc_accept_rate < 1.0);
    REQUIRE(mala_accept_rate > 0.0);
    REQUIRE(mala_accept_rate < 1.0);
    REQUIRE(mh_accept_rate > 0.95); // TODO it rarely rejects; should be 1.0

}

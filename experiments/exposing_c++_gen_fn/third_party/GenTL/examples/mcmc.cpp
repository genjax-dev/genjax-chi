#include <gentl/inference/mcmc.h>
#include <gentl/util/randutils.h>
#include <gentl/types.h>

#include <utility>
#include <vector>
#include <stdexcept>
#include <memory>
#include <array>
#include <iostream>

#include <Eigen/Dense>


// TODO support parameters
// TODO support subset of addresses

using gentl::SimulateOptions;
using gentl::GenerateOptions;
using gentl::UpdateOptions;

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

// model change types

struct ModelChange {
    mean_t new_mean;
    cov_t new_cov;
};

// return value change types

class RetvalChange {};

// learnable parameters

class Parameters { };
class GradientAccumulator { };



class Trace;

class Model {
    typedef int return_type;
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
    std::unique_ptr<Trace> simulate(RNGType& rng, Parameters& parameters, const SimulateOptions&) const;

    // simulate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    void simulate(RNGType& rng, Parameters& parameters, const SimulateOptions&, Trace& trace) const;

    // generate into a new trace object
    template <typename RNGType>
    std::pair<std::unique_ptr<Trace>,double> generate(const EmptyChoiceBuffer& constraints, RNGType& rng,
                                                      Parameters& parameters, const GenerateOptions&) const;

    // generate into an existing trace object (overwriting existing contents)
    template <typename RNGType>
    double generate(Trace& trace, const EmptyChoiceBuffer& constraints,
                    RNGType& rng, Parameters& parameters, const GenerateOptions&) const;

    // equivalent to generate but without returning a trace

    template <typename RNG>
    std::pair<int, double> assess(RNG&, Parameters&, const latent_choices_t& constraints) const;

    template <typename RNG>
    std::pair<int, double> assess(RNG&, Parameters&, const EmptyChoiceBuffer& constraints) const;

};

class Trace {
    friend class Model;
private:
    Model model_;
    double score_;
    latent_choices_t latents_;
    latent_choices_t alternate_latents_;
    latent_choices_t latent_gradient_;
    bool can_be_reverted_;
    bool gradients_computed_;
    RetvalChange diff_;
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
    if (options.precompute_gradient()) {
        logpdf_grad(trace.latent_gradient_, trace.latents_);
    }
    trace.gradients_computed_ = options.precompute_gradient();
    trace.can_be_reverted_ = false;
}

template <typename RNGType>
std::pair<std::unique_ptr<Trace>,double> Model::generate(const EmptyChoiceBuffer& constraints, RNGType& rng,
                                                         Parameters& parameters, const GenerateOptions& options) const {
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
double Model::generate(Trace& trace, const EmptyChoiceBuffer& constraints, RNGType& rng, Parameters& parameters,
                       const GenerateOptions& options) const {
    trace.model_ = *this;
    auto [log_density, log_weight] = importance_sample(trace.latents_, rng);
    trace.score_ = log_density;
    double score = logpdf(trace.latents_);
    if (options.precompute_gradient()) {
        logpdf_grad(trace.latent_gradient_, trace.latents_);
    }
    trace.gradients_computed_ = options.precompute_gradient();
    trace.can_be_reverted_ = false;
    return log_weight;
}

template <typename RNG>
std::pair<int, double> Model::assess(RNG&, Parameters& parameters, const latent_choices_t& constraints) const {
    return {-1, logpdf(constraints)};
}

template <typename RNG>
std::pair<int, double> Model::assess(RNG&, Parameters& parameters, const EmptyChoiceBuffer& constraints) const {
    return {-1, 0.0};
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

const latent_choices_t& Trace::choices() const {
    return latents_;
}

void Trace::revert() {
    if (!can_be_reverted_)
        throw std::logic_error("log_weight is only available between calls to update and revert");
    can_be_reverted_ = false;
    std::swap(latents_, alternate_latents_);
    gradients_computed_ = false;
}


const latent_choices_t& Trace::backward_constraints() {
    return alternate_latents_;
}

const latent_choices_t& Trace::choice_gradient(const LatentsSelection& selection) {
    if (!gradients_computed_) {
        model_.logpdf_grad(latent_gradient_, latents_);
    }
    gradients_computed_ = true;
    return  latent_gradient_;
}

template <typename RNG>
double Trace::update(RNG&, const gentl::change::NoChange&, const latent_choices_t& latents, const UpdateOptions& options) {
    if (options.save()) {
        std::swap(latents_, alternate_latents_);
        latents_ = latents; // copy assignment
        can_be_reverted_ = true;
    } else {
        latents_ = latents; // copy assignment
        // can_be_reverted_ keeps its previous value
    };
    double new_log_density = model_.logpdf(latents_);
    double log_weight = new_log_density - score_;
    score_ = new_log_density;
    if (options.precompute_gradient()) {
        model_.logpdf_grad(latent_gradient_, latents_);
    }
    gradients_computed_ = options.precompute_gradient();
    return log_weight;
}

// TODO add implementation of update that accepts a change to the model


// *********************
// *** Example usage ***
// *********************


int main(int argc, char* argv[]) {
    using std::cout;
    using std::endl;
    using std::cerr;

    // TODO test multiple threads

    // parse arguments

    static const std::string usage = "Usage: ./mcmc"
                                     "<hmc_cycles_per_iter>"
                                     "<mala_cycles_per_iter>"
                                     "<mh_cycles_per_iter>"
                                     "<hmc_leapfrog_steps>"
                                     "<hmc_eps>"
                                     "<mala_tau>"
                                     "<num_threads>"
                                     "<num_iters>"
                                     "<seed>";
    if (argc != 10) {
        throw std::invalid_argument(usage);
    }
    size_t hmc_cycles_per_iter;
    size_t mala_cycles_per_iter;
    size_t mh_cycles_per_iter;
    size_t hmc_leapfrog_steps;
    double hmc_eps;
    double mala_tau;
    size_t num_threads;
    size_t num_iters;
    uint32_t seed;
    try {
        hmc_cycles_per_iter = std::atoi(argv[1]);
        mala_cycles_per_iter = std::atoi(argv[2]);
        mh_cycles_per_iter = std::atoi(argv[3]);
        hmc_leapfrog_steps = std::atoi(argv[4]);
        hmc_eps = std::atof(argv[5]);
        mala_tau = std::atof(argv[6]);
        num_threads = std::atoi(argv[7]);
        num_iters = std::atoi(argv[8]);
        seed = std::atoi(argv[9]);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(usage);
    }
    cerr << "hmc_cycles_per_iter: " << hmc_cycles_per_iter << endl;
    cerr << "mala_cycles_per_iter: " << mala_cycles_per_iter << endl;
    cerr << "mh_cycles_per_iter: " << mh_cycles_per_iter << endl;
    cerr << "hmc_leapfrog_steps: " << hmc_leapfrog_steps << endl;
    cerr << "hmc_eps: " << hmc_eps << endl;
    cerr << "mala_tau: " << mala_tau << endl;
    cerr << "num_threads: " << num_threads << endl;
    cerr << "num_iters: " << num_iters << endl;
    cerr << "seed: " << seed << endl;

    // initialize RNG

    gentl::randutils::seed_seq_fe128 seed_seq {seed};
    std::mt19937 rng(seed_seq);

    // define the model and proposal
    mean_t mean {0.0, 0.0};

    cov_t target_covariance {{1.0, 0.95},
                             {0.95, 1.0}};
    Model model {mean, target_covariance};

    cov_t proposal_covariance {{1.0, 0.0},
                               {0.0, 1.0}};
    Model proposal {mean, proposal_covariance};

    auto make_proposal = [&proposal](const Trace& trace) {
        return proposal;
    };

    // generate initial trace and choice buffers
    Parameters unused {};
    auto [trace, log_weight] = model.generate(EmptyChoiceBuffer{}, rng, unused, GenerateOptions().precompute_gradient(true));
    LatentsSelection hmc_selection;
    LatentsSelection mala_selection;
    auto proposal_trace = make_proposal(*trace).simulate(rng, unused, SimulateOptions().precompute_gradient(false));

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
                    rng, *trace, make_proposal, unused, *proposal_trace, true);
        }
    }

    cerr << "hmc acceptance rate: " << static_cast<double>(hmc_num_accepted) / static_cast<double>(num_iters * hmc_cycles_per_iter) << endl;
    cerr << "mala acceptance rate: " << static_cast<double>(mala_num_accepted) / static_cast<double>(num_iters * mala_cycles_per_iter) << endl;
    cerr << "mh acceptance rate: " << static_cast<double>(mh_num_accepted) / static_cast<double>(num_iters * mh_cycles_per_iter) << endl;

    for (const auto& x : history)
        cout << x(0) << "," << x(1) << endl;

}

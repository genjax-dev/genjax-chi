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

#include <cassert>
#include <iostream>

#include <gentl/learning/supervised.h>
#include <gentl/util/randutils.h>

using gentl::randutils::seed_seq_fe128;
using std::vector;

// y ~ N(m*x+b, 1.0)
// log p(y | x) = -logsqrt(2*pi) - 0.5 * (m*x+b - y)^2
// d log p(y | x) / dm = -(m*x+b - y) * x   = (y - (m*x+b)) * x
// d log p(y | x) / db = -(m*x+b - y)       = y - (m*x+b)

class GradientAccumulator;

class Parameters {
    friend class GradientAccumulator;
    double m_;
    double b_;
    double m_grad_;
    double b_grad_;
public:
    using accumulator_t = GradientAccumulator;
    Parameters(double m, double b) : m_(m), b_(b), m_grad_(0.0), b_grad_(0.0) {}
    double m() const { return m_; }
    double b() const { return b_; }
    double m_grad() const { return m_grad_; }
    double b_grad() const { return b_grad_; }
};

class GradientAccumulator {
    double m_grad_;
    double b_grad_;
    Parameters& parameters_;
public:
    GradientAccumulator(Parameters& parameters) : parameters_(parameters), m_grad_(0.0), b_grad_(0.0) {}
    void add_m_grad(double increment) {
        m_grad_ += increment;
    }
    void add_b_grad(double increment) {
        b_grad_ += increment;
    }
    void update_module_gradients() {
        parameters_.m_grad_ += m_grad_;
        parameters_.b_grad_ += b_grad_;
        m_grad_ = 0.0;
        b_grad_ = 0.0;
    }

};

// return value type
struct Nothing {};
constexpr Nothing nothing;
Nothing zero_gradient(Nothing) { return nothing; }

std::tuple<double,double,double> logpdf_gradient(double x, double y, double m, double b) {
    double m_grad = (y - (m*x+b)) * x;
    double b_grad = y - (m*x+b);
    double x_grad = (y - (m*x+b)) * m;
    return {m_grad, b_grad, x_grad};
}

double logpdf(double x, double y, double m, double b) {
    static double logSqrt2Pi = 0.5*std::log(2*M_PI);
    double z = m*x+b - y;
    return -logSqrt2Pi - 0.5 * z * z;
}

struct Input {
    double x;
    explicit Input(double x_) : x(x_) {}
};

struct Output {
    double y;
};

class Model;
class Trace {
    friend class Model;
    double x_;
    double y_;
    double m_;
    double b_;
    double m_grad_;
    double b_grad_;
    double x_grad_;
private:
    Trace(double x, double y, const Parameters& parameters) : x_(x), y_(y), m_(parameters.m()), b_(parameters.b()) {}
    Trace(double x, double y, const Parameters& parameters, double m_grad, double b_grad, double x_grad) :
            x_(x), y_(y), m_(parameters.m()), b_(parameters.b()), m_grad_(m_grad), b_grad_(b_grad), x_grad_(x_grad) {}
public:
    void parameter_gradient(GradientAccumulator& accumulator, double scaler) {
        std::tie(m_grad_, b_grad_, x_grad_) = logpdf_gradient(x_, y_, m_, b_);
        accumulator.add_m_grad(m_grad_ * scaler);
        accumulator.add_b_grad(b_grad_ * scaler);
    }
    [[nodiscard]] Nothing return_value() const { return nothing; }
};

class Model {
    double x;
public:
    explicit Model(Input input) : x(input.x) {}
    template <typename RNGType>
    std::pair<std::unique_ptr<Trace>,double> generate(RNGType& rng,
                                                      Parameters& parameters, const Output& output,
                                                      const gentl::GenerateOptions& options) {
        double y = output.y;
        double log_weight = logpdf(x, y, parameters.m(), parameters.b());
        if (options.precompute_gradient()) {
            double m_grad, b_grad, x_grad;
            std::tie(m_grad, b_grad, x_grad) = logpdf_gradient(x, y, parameters.m(), parameters.b());
            auto trace = std::unique_ptr<Trace>(new Trace(x, y, parameters, m_grad, b_grad, x_grad));
            return {std::move(trace), log_weight};
        } else {
            auto trace = std::unique_ptr<Trace>(new Trace(x, y, parameters));
            return {std::move(trace), log_weight};
        };
    }
};


TEST_CASE("multi-threaded matches single-threaded", "[supervised]") {

    using datum_type = std::pair<double,double>;
    auto unpack_datum = [](const datum_type& datum) -> std::pair<Model, Output> {
        auto [x, y] = datum;
        Model model{Input{x}};
        Output constraints{y};
        return {model, constraints};
    };

    // random seed
    seed_seq_fe128 minibatch_seed{1};

    // initialize parameters
    double m_init = 0.5;
    double b_init = -0.234;
    Parameters parameters_single{m_init, b_init};
    Parameters parameters_multi{m_init, b_init};

    // training data consists of two points
    auto true_mean = [](double x) { return 2.0 * x + 1.0; };
    std::vector<datum_type> data {{-2.0, true_mean(-2.0)}, {1.123, true_mean(1.123)}};

    auto expected_gradient = [&data,m_init,b_init](const std::vector<size_t>& minibatch) -> std::pair<double,double> {
        double m_grad_expected = 0.0;
        double b_grad_expected = 0.0;
        for (auto idx : minibatch) {
            auto [x, y] = data[idx];
            auto [m_grad_incr, b_grad_incr, x_grad] = logpdf_gradient(x, y, m_init, b_init);
            m_grad_expected += m_grad_incr;//(y - (m_init*x+b_init)) * x;
            b_grad_expected += b_grad_incr;//y - (m_init*x+b_init);
        }
        auto denom = static_cast<double>(minibatch.size());
        m_grad_expected /= denom;
        b_grad_expected /= denom;
        return {m_grad_expected, b_grad_expected};
    };

    bool callback_was_called;
    size_t minibatch_size = 1000;
    std::vector<size_t> minibatch_single;
    std::vector<size_t> minibatch_multi;

    auto callback_single = [&callback_was_called,&minibatch_single](const std::vector<size_t>& minibatch) -> bool {
        callback_was_called = true;
        minibatch_single = minibatch; // copy assignment
        return true; // done (just evaluate the gradient once then return; do not reset gradients)
    };

    auto callback_multi = [&callback_was_called,&minibatch_multi](const std::vector<size_t>& minibatch) -> bool {
        callback_was_called = true;
        minibatch_multi = minibatch; // copy assignment
        return true; // done (just evaluate the gradient once then return; do not reset gradients)
    };

    std::mt19937 rng(minibatch_seed);
    callback_was_called = false;
    gentl::sgd::train_supervised_single_threaded(parameters_single, callback_single, data, unpack_datum, minibatch_size, rng);
    REQUIRE(callback_was_called);

    size_t num_threads = 30;
    callback_was_called = false;
    gentl::sgd::train_supervised(parameters_multi, callback_multi, data, unpack_datum, minibatch_size, num_threads, minibatch_seed);
    REQUIRE(callback_was_called);

    // check that the minibatches are equal (they should use the same the seed)
    REQUIRE(minibatch_single.size() == minibatch_multi.size());
    for (size_t i = 0; i < minibatch_size; i++) {
        REQUIRE(minibatch_single[i] == minibatch_multi[i]);
    }

    // check that they match the expectation
    auto [expected_m_grad, expected_b_grad] = expected_gradient(minibatch_single);
    auto close = [](double a, double b) { return std::abs(a - b) < 1e-10; };
    REQUIRE(close(parameters_single.m_grad(), expected_m_grad));
    REQUIRE(close(parameters_single.b_grad(), expected_b_grad));
    REQUIRE(close(parameters_multi.m_grad(), expected_m_grad));
    REQUIRE(close(parameters_multi.b_grad(), expected_b_grad));

}

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

#include <iostream>

#include <gentl/util/randutils.h>
#include <gentl/inference/sir.h>
#include <gentl/types.h>

using seed_seq_fe128 = gentl::randutils::seed_seq_fe<4, uint32_t>;

struct Trace {
    bool x;
    bool y;
};

struct Parameters {};

struct Constraints {
    bool y;
};

constexpr double prob_x = 0.1;
constexpr double prob_y_given_x = 0.2;
constexpr double prob_y_given_not_x = 0.3;

class Model {
private:
    static double log_prob_y_given_x(bool x, bool y) {
        if (x) {
            if (y) {
                return std::log(prob_y_given_x);
            } else {
                return std::log(1.0 - prob_y_given_x);
            }
        } else {
            if (y) {
                return std::log(prob_y_given_not_x);
            } else {
                return std::log(1.0 - prob_y_given_not_x);
            }
        }
    }
public:

    // generate that allocates new trace
    template <typename RNGType>
    std::pair<std::unique_ptr<Trace>,double> generate(RNGType& rng, const Parameters&, const Constraints& constraints,
                                                      const gentl::GenerateOptions& options) const {
        assert(!options.precompute_gradient());
        std::bernoulli_distribution x_dist{prob_x};
        bool x = x_dist(rng);
        bool y = constraints.y;
        double log_weight = log_prob_y_given_x(x, y);
        return {std::unique_ptr<Trace>(new Trace{x, y}), log_weight};
    }

    // in-place generate
    template <typename RNGType>
    double generate(Trace& trace, RNGType& rng, const Parameters&, const Constraints& constraints,
                    const gentl::GenerateOptions& options) const {
        assert(!options.precompute_gradient());
        std::bernoulli_distribution x_dist{prob_x};
        trace.x = x_dist(rng);
        double log_weight = log_prob_y_given_x(trace.x, constraints.y);
        trace.y = constraints.y;
        return log_weight;
    }
};

TEST_CASE("rolling sampling importance resampling", "[sir]") {

    bool y = true;
    Constraints observation {y};
    seed_seq_fe128 seed_seq{0};
    std::mt19937 rng(seed_seq);
    Model model{};
    Parameters parameters{};
    auto [trace, actual] = gentl::sir::rolling_importance_resampling<Trace>(model, parameters, observation, rng, 1000);
    REQUIRE(trace->y == y);
    double expected = std::log(prob_x * prob_y_given_x + (1.0 - prob_x) * prob_y_given_not_x);
    std::cout << actual << " " << expected << std::endl;
    REQUIRE(std::abs(actual - expected) < 1e-3);

}

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

#ifndef GENTL_SUPERVISED_H
#define GENTL_SUPERVISED_H

#include <cassert>
#include <vector>
#include <utility>
#include <random>
#include <barrier> // NOTE: requires C++20 or c++17 with experimental (consider using boost version or re-implementing)

#include <gentl/util/chunks.h>
#include <gentl/types.h>
#include <gentl/concepts.h>

// TODO implement BBVI and VIMCO...
// TODO implement various learning algorithms like RWS
// TODO provide a callback for generating data on-demand, that defaults to reading from a data set
namespace gentl::sgd {



    template <typename RNGType>
    std::vector<size_t> generate_minibatch(RNGType& rng, size_t dataset_size, size_t minibatch_size) {
        std::vector<size_t> minibatch(minibatch_size);
        std::uniform_int_distribution<size_t> dist {0, dataset_size-1};
        for (int i = 0; i < minibatch_size; i++) {
            minibatch[i] = dist(rng);
        }
        return minibatch;
    }

    template <typename RNGType, typename ParametersType, typename DatumType, typename UnpackDatumType>
    double estimate_objective(RNGType& rng, ParametersType& parameters,
                              const std::vector<DatumType>& data,
                              UnpackDatumType& unpack_datum) {
        double total = 0.0;
        for (const auto& datum : data) {
            auto [model, constraints] = unpack_datum(datum);
            auto trace_and_log_weight = model.generate(rng, parameters, constraints, GenerateOptions());
            total += trace_and_log_weight.second;
        }
        return total / static_cast<double>(data.size());
    }

    template <typename ParametersType, typename RNGType,
            typename StepCallbackType, typename DatasetType, typename UnpackDatumType>
    void train_supervised_single_threaded(ParametersType& parameters,
                                          const StepCallbackType& callback,
                                          const DatasetType& data,
                                          const UnpackDatumType& unpack_datum,
                                          const size_t minibatch_size,
                                          RNGType& rng) {
        typedef typename ParametersType::accumulator_t GradientAccumulatorType;
        GradientAccumulatorType accum {parameters};
        bool done = false;
        const double scaler = 1.0 / static_cast<double>(minibatch_size);
        while (!done) {
            std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);
            for (size_t i = 0; i < minibatch_size; i++) {
                auto [model, constraints] = unpack_datum(data[minibatch[i]]);
                auto [trace, log_weight] = model.generate(rng, parameters, constraints,
                                                          GenerateOptions().precompute_gradient(true));
                //const auto& retval = trace->return_value();
                trace->parameter_gradient(accum, scaler);
            }
            accum.update_module_gradients();
            done = callback(minibatch);
        }
   }

    template <typename ParametersType,
              typename StepCallbackType, typename DatasetType, typename UnpackDatumType,
              typename SeedSequenceType>
    void train_supervised(ParametersType& parameters,
                          const StepCallbackType& callback,
                          const DatasetType& data,
                          const UnpackDatumType& unpack_datum,
                          const size_t minibatch_size,
                          const size_t num_threads,
                          SeedSequenceType& seed) {
        typedef typename ParametersType::accumulator_t GradientAccumulatorType;

        // one gradient accumulator per thread
        std::vector<GradientAccumulatorType> accums;
        for (int i = 0; i < num_threads; i++) {
            accums.emplace_back(GradientAccumulatorType{parameters});
        }

        const size_t data_size = data.size();
        const double scaler = 1.0 / static_cast<double>(minibatch_size);
        size_t iter = 0;
        bool done = false;

        // initialize RNG that will be used to generate minibatches
        std::mt19937 rng(seed);

        std::vector<size_t> minibatch = generate_minibatch(rng, data.size(), minibatch_size);

        auto iteration_serial_stage = [data_size,&iter,&done,&accums,&rng,
                                      &minibatch,&parameters,&data,&callback]() {

            // increments the gradients in the shared parameters object
            // and resets the per-thread gradient accumulators to zero
            for (auto& accum : accums) {
                accum.update_module_gradients();
            }

            // user callback, which implements the gradient step and decides whether we are done or not
            done = callback(minibatch);

            // compute minibatch for next iteration
            if (!done) {
                minibatch = generate_minibatch(rng, data_size, minibatch.size());
            }

            iter++;
        };

        std::barrier sync_point(num_threads, iteration_serial_stage);

        auto stochastic_gradient_chunk = [
                &sync_point,
                &minibatch = std::as_const(minibatch),
                &data = std::as_const(data),
                &parameters,
                scaler,
                &done = std::as_const(done),
                &unpack_datum](GradientAccumulatorType& accum, SeedSequenceType& seed,
                               size_t start, size_t stop) {
            std::mt19937 rng(seed);
            while (!done) {
                for (size_t i = start; i < stop; i++) {
                    auto [model, constraints] = unpack_datum(data[minibatch[i]]);
                    auto [trace, log_weight] = model.generate(rng, parameters, constraints,
                                                              GenerateOptions().precompute_gradient(true));
                    //const auto& retval = trace->return_value();
                    trace->parameter_gradient(accum, scaler);
                }
                sync_point.arrive_and_wait();
            }
        };

        // launch worker threads
        std::vector<std::pair<size_t,size_t>> blocks = even_blocks(minibatch_size, num_threads);
        std::vector<std::thread> threads;
        std::vector<SeedSequenceType> worker_seeds;
        for (size_t i = 0; i < num_threads; i++) {
            auto [start, stop] = blocks[i];
            auto& worker_seed = worker_seeds.emplace_back(seed.spawn());
            threads.emplace_back(stochastic_gradient_chunk, std::ref(accums[i]), std::ref(worker_seed), start, stop);
            start = stop;
        }

        // wait for each worker threads to exit its loop
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

#endif //GENTL_SUPERVISED_H

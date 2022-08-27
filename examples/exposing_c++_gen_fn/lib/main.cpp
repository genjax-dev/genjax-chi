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

// This file exposes the functionality expressed in `gen_fn.cpp`.

#include <gentl/types.h>
#include <gentl/util/randutils.h>
#include <stddef.h>

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gen_fn.h"
#include "pybind11_helpers.h"

using gentl::GenerateOptions;
using gentl::SimulateOptions;

using namespace gen_fn;

// Here, we expose C-like functions to pybind11.
namespace extern_gen_fn {

template <typename T>
void cpu_simulate(void *out, const void **in) {
  const T *mean = reinterpret_cast<const T *>(in[0]);
  const T *cov = reinterpret_cast<const T *>(in[1]);
  unsigned int seed = 314159;
  gentl::randutils::seed_seq_fe128 seed_seq{seed};
  std::mt19937 rng(seed_seq);
  ModelParameters unused{};
  Model model{*mean, *cov};
  auto trace = model.simulate(rng, unused, SimulateOptions());
  T *result = reinterpret_cast<T *>(out);
  *result = trace->choices();
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_simulate_f32"] = EncapsulateFunction(cpu_simulate<float>);
  return dict;
}

PYBIND11_MODULE(expose, m) { m.def("registrations", &Registrations); }

}  // namespace extern_gen_fn

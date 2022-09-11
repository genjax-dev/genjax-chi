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

#ifndef GENTL_MATHUTILS_H
#define GENTL_MATHUTILS_H

namespace gentl::mathutils {

    template<typename T>
    T logsumexp(const std::vector<T> &values) {
        double max = *std::max_element(values.cbegin(), values.cend());
        static T negative_infinity = -std::numeric_limits<T>::infinity();
        if (max == negative_infinity) {
            return negative_infinity;
        } else {
            T sum_exp = 0.0;
            for (auto value: values)
                sum_exp += std::exp(value - max);
            return max + std::log(sum_exp);
        }
    }

    template<typename T>
    T logsumexp(T a, T b) {
        double max = std::max(a, b);
        static T negative_infinity = -std::numeric_limits<T>::infinity();
        if (max == negative_infinity) {
            return negative_infinity;
        } else {
            return max + std::log(1.0 + std::exp(std::min(a, b) - max));
        }
    }

}

#endif //GENTL_MATHUTILS_H

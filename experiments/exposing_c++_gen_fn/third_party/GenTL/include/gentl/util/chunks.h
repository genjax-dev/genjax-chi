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

#ifndef GENTL_CHUNKS_H
#define GENTL_CHUNKS_H

#include <vector>
#include <utility>
#include <cassert>

std::vector<std::pair<size_t,size_t>> even_blocks(size_t num_elements, size_t num_blocks) {
    std::vector<std::pair<size_t,size_t>> blocks {num_blocks};
    size_t start = 0;
    size_t stop;
    for (int i = 0; i < num_blocks; i++) {
        size_t k = num_elements / num_blocks;
        size_t rem = num_elements % num_blocks;
        size_t block_size;
        if (i < rem) {
            block_size = k + 1;
        } else {
            block_size = k;
        }
        stop = start + block_size;
        blocks[i] = {start, stop};
        start = stop;
    }
    assert((*(blocks.end()-1)).second == num_elements);
    return blocks;
}

#endif //GENTL_CHUNKS_H

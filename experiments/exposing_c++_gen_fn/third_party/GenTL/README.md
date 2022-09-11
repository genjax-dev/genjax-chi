# Gen Inference and Learning Template Library (GenTL)
**Work-in-progress** template inference and learning library in C++ based on probabilistic programming principles

[![status](https://github.com/OpenGen/GenTL/actions/workflows/test.yml/badge.svg)](https://github.com/OpenGen/GenTL/actions?query=workflow/test)

[**Documentation (work-in-progress)**](https://opengen.github.io/gentl-docs/latest/)

This is currently a header-only library, and the source code consists of `.h` files under `include/gentl/`.

## Dependencies

### C++ compiler

Most of the library uses C++17 features, which are supported by recent clang++ and g++ versions.

Currently, `include/gentl/learning.h` uses C++20 features, which requires g++ version 11 or above, and clang++ version 11 or above.

### CMake

CMake 3.7 or above. See [Installing CMake](https://cmake.org/install/).

### Doxygen

This is only required for generating documentation.

See [Doxygen Installation](https://www.doxygen.nl/manual/install.html).

## Testing

Use CMake to build and test. Note that you can configure the C++ compiler to use with either the `CXX` environment variable or by setting the `CMAKE_CXX_COMPILER` variable, e.g.:
```
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-11
cmake --build build
cmake --build build --target test
```

## Generating documentation

```
cmake --build --target docs
```
Note that there is a Github Action that generates and deploys documentation to https://github.com/OpenGen/gentl-docs/ when there is a `push` to the `main` branch of this repository.


## References

Gen: A General-Purpose Probabilistic Programming System with Programmable Inference. Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI ‘19). ([pdf](https://dl.acm.org/doi/10.1145/3314221.3314642)) ([bibtex](assets/gen-pldi.txt))

Marco Cusumano-Towner's [PhD thesis](https://www.mct.dev).

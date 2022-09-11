set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION "MathJax_3")
set(DOXYGEN_MATHJAX_EXTENSIONS "ams")

doxygen_add_docs(docs
        gentl/concepts.h
        gentl/types.h
        gentl/modeling/sequence.h
        gentl/inference/mcmc.h
        gentl/inference/sir.h
        gentl/inference/particle_filter.h
        gentl/learning/supervised.h
        gentl/util/randutils.h
        gentl/util/mathutils.h
        "${PROJECT_SOURCE_DIR}/docs/mainpage.md"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")

# TODO add sphinx on top of Doxygen, as described here:
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
# actually, find the C++ library that has really nie docs

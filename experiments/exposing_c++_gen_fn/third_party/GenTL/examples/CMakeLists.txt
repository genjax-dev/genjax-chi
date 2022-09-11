add_executable(mcmc mcmc.cpp)
target_include_directories(mcmc PRIVATE ../include)
target_include_directories(mcmc PRIVATE ../third_party/Eigen)

add_executable(particle_filter particle_filter.cpp)
target_include_directories(particle_filter PRIVATE ../include)
target_include_directories(particle_filter PRIVATE ../third_party/Eigen)

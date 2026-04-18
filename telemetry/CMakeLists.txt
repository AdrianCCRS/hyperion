cmake_minimum_required(VERSION 3.18)
project(telemetry CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
# Release with debug info for profiling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
 
add_compile_options(-Wall -Wextra -O2 -march=native)
 
add_library(telemetry STATIC
  src/perf_reader.cpp
  src/rapl_reader.cpp
  src/nvml_reader.cpp
  src/collector.cpp
)
target_include_directories(telemetry PUBLIC include)
target_link_libraries(telemetry PRIVATE pthread)
 
# Link NVML only when GPU support is requested
option(WITH_GPU "Enable NVML GPU telemetry" OFF)
if(WITH_GPU)
  find_library(NVML_LIB nvidia-ml REQUIRED)
  target_link_libraries(telemetry PRIVATE ${NVML_LIB})
  target_compile_definitions(telemetry PUBLIC TELEMETRY_WITH_GPU)
endif()

enable_testing()

add_executable(telemetry_link_smoke_test
  tests/test_link_smoke.cpp
)
target_link_libraries(telemetry_link_smoke_test PRIVATE telemetry pthread)
add_test(NAME telemetry_link_smoke_test COMMAND telemetry_link_smoke_test)

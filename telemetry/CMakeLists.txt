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
  src/perf_cgroup_reader.cpp
  src/rapl_reader.cpp
  src/nvml_reader.cpp
  src/collector.cpp
  src/experiment_utils.cpp
)
target_include_directories(telemetry PUBLIC include)
target_link_libraries(telemetry PRIVATE pthread)

add_executable(telemetry_overhead_bench
  benchmarks/telemetry_overhead_bench.cpp
)
target_link_libraries(telemetry_overhead_bench PRIVATE telemetry pthread)

add_executable(telemetry_jitter_bench
  benchmarks/telemetry_jitter_bench.cpp
)
target_link_libraries(telemetry_jitter_bench PRIVATE telemetry pthread)

add_executable(telemetry_kernel_workload
  experiments/telemetry_kernel_workload.cpp
)
target_link_libraries(telemetry_kernel_workload PRIVATE telemetry pthread)

add_executable(telemetry_kernel_launcher
  experiments/telemetry_kernel_launcher.cpp
)
target_link_libraries(telemetry_kernel_launcher PRIVATE telemetry pthread)
 
# Link NVML only when GPU support is requested
option(WITH_GPU "Enable NVML GPU telemetry" OFF)
if(WITH_GPU)
  find_path(NVML_INCLUDE_DIR nvml.h)
  find_library(NVML_LIB nvidia-ml REQUIRED)
  if(NOT NVML_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find nvml.h")
  endif()
  target_include_directories(telemetry PUBLIC ${NVML_INCLUDE_DIR})
  target_link_libraries(telemetry PRIVATE ${NVML_LIB})
  target_compile_definitions(telemetry PUBLIC TELEMETRY_WITH_GPU)
endif()

enable_testing()

add_executable(telemetry_link_smoke_test
  tests/test_link_smoke.cpp
)
target_link_libraries(telemetry_link_smoke_test PRIVATE telemetry pthread)
add_test(NAME telemetry_link_smoke_test COMMAND telemetry_link_smoke_test)

add_executable(spsc_ring_test
  tests/test_spsc_ring.cpp
)
target_link_libraries(spsc_ring_test PRIVATE telemetry pthread)
add_test(NAME spsc_ring_test COMMAND spsc_ring_test)

add_executable(collector_lifecycle_test
  tests/test_collector_lifecycle.cpp
)
target_link_libraries(collector_lifecycle_test PRIVATE telemetry pthread)
add_test(NAME collector_lifecycle_test COMMAND collector_lifecycle_test)

add_executable(collector_no_perf_test
  tests/test_collector_no_perf.cpp
)
target_link_libraries(collector_no_perf_test PRIVATE telemetry pthread)
add_test(NAME collector_no_perf_test COMMAND collector_no_perf_test)

add_executable(perf_reader_test
  tests/test_perf_reader.cpp
)
target_link_libraries(perf_reader_test PRIVATE telemetry pthread)
add_test(NAME perf_reader_test COMMAND perf_reader_test)

add_executable(perf_cgroup_reader_test
  tests/test_perf_cgroup_reader.cpp
)
target_link_libraries(perf_cgroup_reader_test PRIVATE telemetry pthread)
add_test(NAME perf_cgroup_reader_test COMMAND perf_cgroup_reader_test)

add_executable(rapl_reader_test
  tests/test_rapl_reader.cpp
)
target_link_libraries(rapl_reader_test PRIVATE telemetry pthread)
add_test(NAME rapl_reader_test COMMAND rapl_reader_test)

add_executable(experiment_utils_test
  tests/test_experiment_utils.cpp
)
target_link_libraries(experiment_utils_test PRIVATE telemetry pthread)
add_test(NAME experiment_utils_test COMMAND experiment_utils_test)

if(NOT WITH_GPU)
  add_executable(nvml_reader_cpu_stub_test
    tests/test_nvml_reader_cpu_stub.cpp
  )
  target_link_libraries(nvml_reader_cpu_stub_test PRIVATE telemetry pthread)
  add_test(NAME nvml_reader_cpu_stub_test COMMAND nvml_reader_cpu_stub_test)
endif()

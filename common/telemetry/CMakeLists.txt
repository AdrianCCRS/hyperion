# Build genérico para Rocky Linux 9 y Fedora. En un nodo limpio de
# cualquiera de las dos distros, el caso normal es:
#   dnf install gcc-c++ cmake make            # (Rocky: primero 'dnf config-manager
#                                              #  --set-enabled crb' para el grupo de
#                                              #  desarrollo; Fedora lo trae por defecto)
#   cmake -S . -B build -DWITH_GPU=ON
#   cmake --build build -j
# Sin más flags: CMake ya encuentra un g++ normal del sistema como CXX y,
# con el driver NVIDIA propietario instalado, encuentra nvml.h y
# libnvidia-ml.so por las rutas estándar de la distro.
#
# Dos situaciones que sí requieren declarar flags a mano (ambas ya vistas
# en clústeres HPC reales que cargan compiladores/CUDA vía un sistema de
# módulos en vez del paquete de la distro, p. ej. OpenHPC/Lmod):
#   1. Si el `module load` activo del clúster antepone al PATH un
#      compilador no estándar (p. ej. nvc++ del NVIDIA HPC SDK) en vez de
#      un g++ real, ese compilador puede no traer una libstdc++ completa
#      (falla en tiempo de ejecución con "GLIBCXX_x.y.z not found" aunque
#      compile sin error). Nada de este harness requiere un compilador de
#      dispositivo CUDA (no hay archivos .cu aquí) — fuerce un g++/clang++
#      real con -DCMAKE_CXX_COMPILER=/ruta/a/g++.
#   2. Si el módulo CUDA del clúster solo instala la librería NVML
#      versionada del driver (`libnvidia-ml.so.1`, sin el symlink
#      `.so` sin versión que `find_library` espera) o el `nvml.h` no está
#      en una ruta estándar, apunte ambos a mano:
#      -DNVML_LIB=/ruta/a/libnvidia-ml.so(.1) -DNVML_INCLUDE_DIR=/ruta/a/include
#
# Nota de portabilidad importante: -march=native (abajo) compila para la
# CPU exacta de la máquina donde se ejecuta `cmake --build`. Si se compila
# en un nodo de login y se ejecuta en un nodo de cómputo con una
# microarquitectura distinta, el binario puede fallar con "illegal
# instruction" o (peor) medir un P_pico de FLOPs incorrecto sin fallar.
# Compilar siempre en el mismo tipo de nodo donde correrá la campaña real
# (o dentro de la misma asignación de Slurm/cgroup), nunca en un login
# node genérico compartido entre arquitecturas distintas.
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
  src/uncore_reader.cpp
  src/cpu_freq_reader.cpp
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

add_executable(perf_reader_pid_live_test
  tests/test_perf_reader_pid_live.cpp
)
target_link_libraries(perf_reader_pid_live_test PRIVATE telemetry pthread)
add_test(NAME perf_reader_pid_live_test COMMAND perf_reader_pid_live_test)
set_tests_properties(perf_reader_pid_live_test PROPERTIES SKIP_RETURN_CODE 77)

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

add_executable(uncore_reader_test
  tests/test_uncore_reader.cpp
)
target_link_libraries(uncore_reader_test PRIVATE telemetry pthread)
add_test(NAME uncore_reader_test COMMAND uncore_reader_test)

add_executable(cpu_freq_reader_test
  tests/test_cpu_freq_reader.cpp
)
target_link_libraries(cpu_freq_reader_test PRIVATE telemetry pthread)
add_test(NAME cpu_freq_reader_test COMMAND cpu_freq_reader_test)

add_executable(experiment_utils_test
  tests/test_experiment_utils.cpp
)
target_link_libraries(experiment_utils_test PRIVATE telemetry pthread)
add_test(NAME experiment_utils_test COMMAND experiment_utils_test)

add_executable(gpu_clock_controller_test
  tests/test_gpu_clock_controller.cpp
)
target_link_libraries(gpu_clock_controller_test PRIVATE telemetry pthread)
add_test(NAME gpu_clock_controller_test COMMAND gpu_clock_controller_test)

add_executable(collector_gpu_cadence_test
  tests/test_collector_gpu_cadence.cpp
)
target_link_libraries(collector_gpu_cadence_test PRIVATE telemetry pthread)
add_test(NAME collector_gpu_cadence_test COMMAND collector_gpu_cadence_test)
set_tests_properties(collector_gpu_cadence_test PROPERTIES SKIP_RETURN_CODE 77)

if(NOT WITH_GPU)
  add_executable(nvml_reader_cpu_stub_test
    tests/test_nvml_reader_cpu_stub.cpp
  )
  target_link_libraries(nvml_reader_cpu_stub_test PRIVATE telemetry pthread)
  add_test(NAME nvml_reader_cpu_stub_test COMMAND nvml_reader_cpu_stub_test)
endif()

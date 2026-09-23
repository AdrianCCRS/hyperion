# Daemon de GPU en C++ (Bloque D). Reemplaza a fase3_daemon/run_daemon.py (Python) por decision
# del usuario (2026-09-23), igual que el loop de CPU (fase3_daemon/cpu_loop). Los cuatro tests
# del nucleo no tienen dependencias externas y siempre se construyen; `gpu_loop_main` necesita
# ONNX Runtime y NVML:
#   cmake -S fase3_daemon/gpu_loop_cpp -B fase3_daemon/gpu_loop_cpp/build \
#     -DONNXRUNTIME_ROOT=$HOME/.conda/envs/hyperion-cpu-onnx \
#     -DNVML_INCLUDE_DIR=<cuda>/include -DNVML_LIB=<cuda>/targets/x86_64-linux/lib/stubs/libnvidia-ml.so
# (NVML se enlaza contra el stub del toolkit; en ejecucion el cargador resuelve libnvidia-ml.so.1 del driver.)
cmake_minimum_required(VERSION 3.18)
project(hyperion_gpu_loop CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -O2)

find_package(Threads REQUIRED)
add_library(gpu_loop_cpp INTERFACE)
target_include_directories(gpu_loop_cpp INTERFACE
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_loop/include            # decision_log.hpp (mismo esquema JSONL que CPU)
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/telemetry/include) # gpu_clock_controller.hpp
target_link_libraries(gpu_loop_cpp INTERFACE Threads::Threads)

enable_testing()
foreach(t gpu_activity_tracker gpu_window_classifier gpu_active_writer gpu_clock_actuator)
  add_executable(${t}_test tests/test_${t}.cpp)
  target_include_directories(${t}_test PRIVATE tests)
  target_link_libraries(${t}_test PRIVATE gpu_loop_cpp)
  add_test(NAME ${t}_test COMMAND ${t}_test)
endforeach()

if(DEFINED ONNXRUNTIME_ROOT AND NVML_INCLUDE_DIR AND NVML_LIB)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
  find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib)
endif()
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB AND NVML_INCLUDE_DIR AND NVML_LIB)
  message(STATUS "ONNX Runtime + NVML encontrados -- construyendo gpu_loop_main")
  add_executable(gpu_loop_main src/gpu_loop_main.cpp)
  target_include_directories(gpu_loop_main PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${NVML_INCLUDE_DIR})
  target_link_directories(gpu_loop_main PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(gpu_loop_main PRIVATE gpu_loop_cpp ${ONNXRUNTIME_LIB} ${NVML_LIB})
  # mismo motivo que cpu_loop: el libstdc++ de conda es mas nuevo que el del sistema
  set_target_properties(gpu_loop_main PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
else()
  message(STATUS "gpu_loop_main NO se construye: falta ONNXRUNTIME_ROOT o NVML_INCLUDE_DIR/NVML_LIB")
endif()

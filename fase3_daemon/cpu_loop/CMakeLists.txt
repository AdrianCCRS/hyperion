# Loop de CPU del daemon (§4.3 punto 5 del plan de realineación).
#
# Bloque A (2026-09-22): xgboost_cpu.onnx ya existe y está verificado
# (export_onnx.py, fila a fila contra 1.17M filas reales). Bloque B añade
# aquí lo que faltaba: cpu_feature_builder.hpp (deltas -> 6 variables,
# header-only, sin dependencias externas) y onnx_cpu_classifier.hpp
# (envoltorio de ONNX Runtime C++). AMBOS se prueban con datos/tensores
# sintéticos, sin `collector.hpp` en vivo ni permisos de PMU reales -- el
# tick real de ~1ms leyendo collector.hpp sigue pendiente (falta medir la
# latencia de inferencia de este bloque primero, §4.3: "medir antes de
# integrar al camino caliente").
#
# ONNX Runtime C++ NO viene con el sistema ni con conda base -- se instaló
# en un entorno conda-forge dedicado y liviano (sin CUDA, la inferencia de
# CPU no la necesita):
#   conda create -n hyperion-cpu-onnx -c conda-forge cmake onnxruntime-cpp onnx
# Build:
#   cmake -S fase3_daemon/cpu_loop -B fase3_daemon/cpu_loop/build \
#     -DONNXRUNTIME_ROOT=$(conda info --base)/envs/hyperion-cpu-onnx
#   cmake --build fase3_daemon/cpu_loop/build
# Sin -DONNXRUNTIME_ROOT (o si no se encuentra ahí), los targets que
# dependen de ONNX Runtime simplemente no se registran -- cpu_phase_controller_test
# sigue construyéndose y corriendo igual, como hasta ahora.
cmake_minimum_required(VERSION 3.18)
project(hyperion_cpu_loop CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -O2)

add_library(cpu_loop INTERFACE)
target_include_directories(cpu_loop INTERFACE include)
# Solo para telemetry/metrics.hpp (CpuSample) -- header-only, no se enlaza
# la biblioteca estática `telemetry` completa (esa arrastra perf_reader.cpp
# y compañía, con sus propias dependencias de sistema/PMU que este target
# no necesita para probar el cálculo de features de forma aislada).
target_include_directories(cpu_loop INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../common/telemetry/include)

enable_testing()
add_executable(cpu_phase_controller_test tests/test_cpu_phase_controller.cpp)
target_link_libraries(cpu_phase_controller_test PRIVATE cpu_loop)
add_test(NAME cpu_phase_controller_test COMMAND cpu_phase_controller_test)

add_executable(cpu_feature_builder_test tests/test_cpu_feature_builder.cpp)
target_link_libraries(cpu_feature_builder_test PRIVATE cpu_loop)
add_test(NAME cpu_feature_builder_test COMMAND cpu_feature_builder_test)

add_executable(cpu_loop_tick_test tests/test_cpu_loop_tick.cpp)
target_link_libraries(cpu_loop_tick_test PRIVATE cpu_loop)
add_test(NAME cpu_loop_tick_test COMMAND cpu_loop_tick_test)

if(DEFINED ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
  find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib)
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
  message(STATUS "ONNX Runtime encontrado: ${ONNXRUNTIME_LIB} -- construyendo targets de inferencia")
  # libonnxruntime.so de conda-forge se compila contra un libstdc++ mas
  # nuevo (GLIBCXX_3.4.34) que el del sistema -- sin esto, el enlazador
  # resuelve simbolos contra /usr/lib/.../libstdc++.so.6 (mas viejo) antes
  # que el de este mismo entorno conda, y falla con "undefined reference"
  # pese a que la version correcta SI esta instalada. rpath ademas de -L:
  # el binario debe encontrar la version correcta tambien en tiempo de
  # ejecucion, no solo al enlazar.
  add_executable(onnx_cpu_classifier_test tests/test_onnx_cpu_classifier.cpp)
  target_include_directories(onnx_cpu_classifier_test PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_directories(onnx_cpu_classifier_test PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(onnx_cpu_classifier_test PRIVATE cpu_loop ${ONNXRUNTIME_LIB})
  set_target_properties(onnx_cpu_classifier_test PROPERTIES
                         BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
  add_test(NAME onnx_cpu_classifier_test COMMAND onnx_cpu_classifier_test
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(cpu_loop_latency_bench tests/latency_bench.cpp)
  target_include_directories(cpu_loop_latency_bench PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_directories(cpu_loop_latency_bench PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(cpu_loop_latency_bench PRIVATE cpu_loop ${ONNXRUNTIME_LIB})
  set_target_properties(cpu_loop_latency_bench PROPERTIES
                         BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
else()
  message(STATUS "ONNX Runtime NO encontrado (pasa -DONNXRUNTIME_ROOT=<prefijo conda>) -- "
                  "se omiten onnx_cpu_classifier_test y cpu_loop_latency_bench")
endif()

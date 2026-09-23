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
# cpu_freq_actuator.hpp usa std::thread (escrituras en paralelo)
find_package(Threads REQUIRED)
target_link_libraries(cpu_loop INTERFACE Threads::Threads)
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

# cpu_loop_consumer.hpp usa telemetry::Collector::Ring (para el tipo, no
# para construir un Collector real) -- basta con el include header-only ya
# agregado arriba, sin enlazar la biblioteca telemetry ni PMU real: el test
# empuja muestras sinteticas directo al ring con try_push().
add_executable(cpu_loop_consumer_test tests/test_cpu_loop_consumer.cpp)
target_link_libraries(cpu_loop_consumer_test PRIVATE cpu_loop)
add_test(NAME cpu_loop_consumer_test COMMAND cpu_loop_consumer_test)

# decision_log.hpp no depende de ONNX Runtime ni de PMU -- se prueba
# siempre, igual que el resto del nucleo (Bloque C, registro estructurado
# por decision, Plan_Fase3_Daemon.md SS0.1 requisito 2).
add_executable(decision_log_test tests/test_decision_log.cpp)
target_link_libraries(decision_log_test PRIVATE cpu_loop)
add_test(NAME decision_log_test COMMAND decision_log_test)

# Senal de coordinacion CPU-GPU (Bloque C, item C4) -- sin dependencias
# externas, siempre se construye y se prueba.
add_executable(gpu_active_reader_test tests/test_gpu_active_reader.cpp)
target_link_libraries(gpu_active_reader_test PRIVATE cpu_loop)
add_test(NAME gpu_active_reader_test COMMAND gpu_active_reader_test)

# Sonda real para verify_gpu_coordination_e2e.py (Bloque C, item C4) --
# proceso C++ separado real, sin PMU/ONNX, solo para probar la señal.
add_executable(gpu_active_signal_probe tools/gpu_active_signal_probe.cpp)
target_link_libraries(gpu_active_signal_probe PRIVATE cpu_loop pthread)

# Actuador nativo de frecuencia (Bloque C8) -- sin dependencias externas,
# probado sobre un sysfs simulado (sin hardware ni permisos), siempre se
# construye. La sonda es para tests/test_freqctl_parity.py (paridad con
# common/hpc/freqctl.py).
add_executable(cpu_freq_actuator_test tests/test_cpu_freq_actuator.cpp)
target_link_libraries(cpu_freq_actuator_test PRIVATE cpu_loop)
add_test(NAME cpu_freq_actuator_test COMMAND cpu_freq_actuator_test)

add_executable(cpu_freq_actuator_probe tools/cpu_freq_actuator_probe.cpp)
target_link_libraries(cpu_freq_actuator_probe PRIVATE cpu_loop)
add_test(NAME freqctl_parity_test
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_freqctl_parity.py $<TARGET_FILE:cpu_freq_actuator_probe>)

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

  # cpu_loop_main.cpp -- el binario de produccion real (§4.3, equivalente a
  # run_daemon.py del lado GPU). Necesita ADEMAS la biblioteca `telemetry`
  # completa (perf_reader.cpp y compania -- PMU real), no solo sus headers.
  # Requiere permisos de PMU para correr (nunca funciona sin ellos, ni en
  # pacca fuera de un nodo de computo real) -- se compila aqui porque
  # compilar no requiere esos permisos, solo ejecutarlo los requiere.
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../common/telemetry
                    ${CMAKE_CURRENT_BINARY_DIR}/telemetry_build)
  add_executable(cpu_loop_main src/cpu_loop_main.cpp)
  target_include_directories(cpu_loop_main PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_directories(cpu_loop_main PRIVATE ${ONNXRUNTIME_ROOT}/lib)
  target_link_libraries(cpu_loop_main PRIVATE cpu_loop telemetry pthread ${ONNXRUNTIME_LIB})
  set_target_properties(cpu_loop_main PROPERTIES
                         BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
else()
  message(STATUS "ONNX Runtime NO encontrado (pasa -DONNXRUNTIME_ROOT=<prefijo conda>) -- "
                  "se omiten onnx_cpu_classifier_test y cpu_loop_latency_bench")
endif()

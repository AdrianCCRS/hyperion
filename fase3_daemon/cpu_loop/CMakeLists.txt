# Loop de CPU del daemon (§4.3 punto 5 del plan de realineación). Solo la
# máquina de decisión (cpu_phase_controller.hpp) vive aquí por ahora --
# header-only, sin dependencia de common/telemetry ni de un runtime de
# inferencia real, a propósito (mismo principio que gpu_clock_controller.hpp:
# probable sin hardware, sin ONNX Runtime, sin modelo entrenado real).
#
# NO INCLUIDO TODAVÍA (bloqueado por piezas que no existen en este entorno
# de reconstrucción, documentado en fase3_daemon/README.md):
#   - Carga/inferencia del modelo de fase2_clasificador/ en C++ (requiere
#     exportar a ONNX + ONNX Runtime C++ SDK, ninguno de los dos disponible
#     aquí -- no hay onnx/skl2onnx instalados, y ONNX Runtime C++ headers
#     no están en este entorno, solo el paquete Python).
#   - El tick real de ~1ms leyendo common/telemetry/include/telemetry/
#     collector.hpp en vivo.
# cpu_phase_controller.hpp es el punto de extensión: ambas piezas, cuando
# existan, llaman a CpuPhaseController::on_window() con la clase ya
# inferida -- no hace falta tocar este archivo para añadirlas.
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

enable_testing()
add_executable(cpu_phase_controller_test tests/test_cpu_phase_controller.cpp)
target_link_libraries(cpu_phase_controller_test PRIVATE cpu_loop)
add_test(NAME cpu_phase_controller_test COMMAND cpu_phase_controller_test)

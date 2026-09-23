/**
 * @file
 * @brief Sonda mínima para verificar de punta a punta la señal de
 * coordinación CPU-GPU (Bloque C, ítem C4): lee `--signal-path` cada
 * `--interval-ms` durante `--duration-s` y escribe una línea
 * `ts_ns,active` por lectura a stdout. Sin PMU ni ONNX -- no es
 * `cpu_loop_main`, es un proceso C++ real y separado que solo ejercita
 * `GpuActiveFileReader`, para poder correr concurrentemente con un
 * escritor Python real (`verify_gpu_coordination_e2e.py`) sin necesitar
 * permisos de PMU para probar esto específicamente.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "gpu_active_reader.hpp"

int main(int argc, char** argv) {
    std::string signal_path;
    double duration_s = 5.0;
    int interval_ms = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requiere un valor\n", flag); std::exit(2); }
            return argv[++i];
        };
        if (arg == "--signal-path") signal_path = need("--signal-path");
        else if (arg == "--duration-s") duration_s = std::stod(need("--duration-s"));
        else if (arg == "--interval-ms") interval_ms = std::stoi(need("--interval-ms"));
        else { std::fprintf(stderr, "flag desconocida: %s\n", arg.c_str()); std::exit(2); }
    }
    if (signal_path.empty()) { std::fprintf(stderr, "--signal-path es obligatorio\n"); std::exit(2); }

    hyperion::cpu_loop::GpuActiveFileReader reader(signal_path);
    const auto deadline = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(static_cast<long>(duration_s * 1000));
    while (std::chrono::steady_clock::now() < deadline) {
        const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
        std::printf("%lld,%d\n", static_cast<long long>(now_ns), reader.read_active() ? 1 : 0);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    return 0;
}

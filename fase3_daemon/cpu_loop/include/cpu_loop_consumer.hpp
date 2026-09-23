#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <thread>

#include "cpu_loop_tick.hpp"
#include "telemetry/collector.hpp"

/**
 * @file
 * @brief Drena `telemetry::Collector::Ring` en vivo y ejecuta un
 * `run_cpu_tick()` por cada par consecutivo de muestras CPU (§4.3 punto 5).
 *
 * No construye ni posee el `Ring` -- mismo principio de "esta pieza no
 * descubre nada por su cuenta" que el resto del daemon. El llamador decide
 * si el ring lo alimenta un `telemetry::Collector` real (producción,
 * `cpu_loop_main.cpp`, requiere PMU real -> solo en pacca) o `try_push()`
 * manual de muestras sintéticas (pruebas, sin PMU ni modelo real -- ver
 * `tests/test_cpu_loop_consumer.cpp`, corrible en cualquier máquina).
 *
 * `gpu_active` se resuelve por una función inyectada, no por una variable
 * atómica compartida como describe el plan (§4.1: "una variable atómica
 * compartida entre ambos loops... viven en el mismo proceso"). Ese supuesto
 * ya no se sostiene tal cual: el loop de GPU es Python (`run_daemon.py`) y
 * este es un binario C++ separado, dos procesos distintos. Mientras no
 * exista un mecanismo de coordinación entre procesos definido (candidato:
 * un archivo de estado que el loop de GPU escribe y este lee cada tick,
 * o un socket local), `gpu_active` es una función inyectable que hoy puede
 * ser tan simple como `[]{ return false; }` -- decisión de alcance
 * explícita, no un hueco escondido: como la política de GPU en
 * `memory_bound` sigue bloqueada por H1 (Plan_Fase3_Daemon.md Bloque D),
 * no hay todavía un escenario real donde el loop de GPU esté aplicando
 * reloj y este loop necesite coordinarse con él.
 */
namespace hyperion::cpu_loop {

using GpuActiveFn = std::function<bool()>;
using TickObserver = std::function<void(const TickResult&)>;

/**
 * @brief Bucle de consumo. Corre hasta que `stop` se active.
 *
 * @param ring Ring ya construido (y, en producción, ya conectado a un
 *   `telemetry::Collector` con `start()` llamado por el caller).
 * @param stop Bandera de parada, consultada cada iteración -- el llamador
 *   la activa desde un manejador de señal (SIGINT/SIGTERM) o, en pruebas,
 *   tras drenar un número conocido de muestras.
 * @param predict_proba Inferencia inyectada (en producción,
 *   `OnnxCpuClassifier::predict_memory_bound_proba`).
 * @param gpu_active Resuelve la barrera de coordinación CPU-GPU en cada
 *   tick (ver docstring del archivo).
 * @param on_tick Callback de observación -- logging estructurado (§4.3
 *   punto 10) o, en pruebas, acumular resultados en un vector.
 * @param idle_sleep Espera cuando el ring está vacío, mismo patrón que
 *   `telemetry_kernel_launcher.cpp::drain_samples` (100us por defecto).
 */
inline void run_consumer_loop(
    telemetry::Collector::Ring& ring, std::atomic<bool>& stop,
    const PredictProbaFn& predict_proba, float threshold,
    const GpuActiveFn& gpu_active, CpuPhaseController& controller,
    const TickObserver& on_tick,
    std::chrono::nanoseconds idle_sleep = std::chrono::microseconds(100)
) {
    std::optional<telemetry::CpuSample> prev;

    const auto drain_once = [&]() -> bool {
        bool drained_any = false;
        while (auto sample = ring.try_pop()) {
            drained_any = true;
            if (sample->tag != telemetry::SampleTag::CPU) continue;
            if (prev) {
                const int64_t delta_t_ns =
                    static_cast<int64_t>(sample->cpu.timestamp_ns) - static_cast<int64_t>(prev->timestamp_ns);
                on_tick(run_cpu_tick(*prev, sample->cpu, delta_t_ns, predict_proba, threshold,
                                     gpu_active(), controller));
            }
            prev = sample->cpu;
        }
        ring.flush_consumer();
        return drained_any;
    };

    while (!stop.load(std::memory_order_relaxed)) {
        if (!drain_once()) {
            std::this_thread::sleep_for(idle_sleep);
        }
    }
    drain_once();  // drenaje final tras stop, no perder lo ya empujado al ring
}

}  // namespace hyperion::cpu_loop

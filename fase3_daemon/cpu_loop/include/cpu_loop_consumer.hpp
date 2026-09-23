#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
 * este es un binario C++ separado, dos procesos distintos. Bloque C, ítem
 * C4: el mecanismo de coordinación entre procesos ya existe
 * (`gpu_active_reader.hpp` / `fase3_daemon/gpu_loop/coordination.py`, un
 * archivo de un byte reemplazado atómicamente en cada transición de fase)
 * -- `cpu_loop_main.cpp` lo conecta con `--gpu-active-signal-path`. Sigue
 * siendo una función inyectable aquí (para poder probarse con
 * `[]{ return false; }` u otra fuente, ver `tests/test_cpu_loop_consumer.cpp`),
 * y sin esa bandera el binario de producción mantiene `false` siempre --
 * el único comportamiento con sentido mientras la política GPU en
 * `memory_bound` siga bloqueada por H1 (Plan_Fase3_Daemon.md Bloque D): no
 * hay todavía un escenario real donde el loop de GPU esté aplicando reloj.
 */
namespace hyperion::cpu_loop {

using GpuActiveFn = std::function<bool()>;
using TickObserver = std::function<void(const TickResult&)>;

/** Tope del drenaje final tras `stop` (ver run_consumer_loop). */
constexpr size_t kFinalDrainMaxItems = 64;  // a ~28 ms por escritura, 64 ticks ya son ~2 s de apagado

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

    // `honor_stop`: dentro del drenaje se consulta `stop` en CADA muestra. Antes
    // solo se miraba entre drenajes, y si el consumidor iba mas lento que el
    // productor (p.ej. una escritura de frecuencia de ~28 ms por tick) el ring
    // nunca se vaciaba y SIGTERM/SIGINT no detenian el proceso, con lo que la
    // restauracion por senal no corria (preflight C8, job 7592). El drenaje
    // final tras `stop` se acota a `max_items` por la misma razon: el productor
    // sigue empujando hasta que el llamador detiene el collector.
    const auto drain_once = [&](bool honor_stop, size_t max_items) -> bool {
        bool drained_any = false;
        size_t n = 0;
        while ((!honor_stop || !stop.load(std::memory_order_relaxed)) && n < max_items) {
            auto sample = ring.try_pop();
            if (!sample) break;
            ++n;
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
        if (!drain_once(/*honor_stop=*/true, SIZE_MAX)) {
            std::this_thread::sleep_for(idle_sleep);
        }
    }
    drain_once(/*honor_stop=*/false, kFinalDrainMaxItems);  // drenaje final acotado, ver arriba
}

}  // namespace hyperion::cpu_loop

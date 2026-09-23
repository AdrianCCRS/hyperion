#pragma once
#include <cstdint>

/**
 * @file
 * @brief Detección de fase de GPU por actividad (Opción C del plan: sondeo de
 * NVML), en C++. Puerto de `fase3_daemon/gpu_loop/activity_poller.py`
 * (`poll_phase_events`): mismas reglas, sin el generador ni el sondeo, que
 * ahora vive en el bucle de `gpu_loop_main` (NVML directo, sin subprocesos).
 *
 * Semántica preservada:
 *  - Una fase empieza cuando `util` cruza el umbral de abajo hacia arriba y
 *    termina cuando vuelve por debajo.
 *  - `min_active_ns` > 0: la DECISIÓN se emite solo tras esa ventana de
 *    actividad sostenida (una actividad que cae antes no genera decisión,
 *    pero sí `ended`). Con 0, en la primera muestra activa (comportamiento
 *    original, que clasifica con la muestra instantánea del flanco de
 *    subida; medido: acierta 2 de 6, job 7606).
 *  - Un solo `decide` por fase activa.
 * Orden que debe respetar el llamador en cada muestra: si `active_start`,
 * vaciar la ventana del clasificador ANTES de registrar la muestra; registrar
 * la muestra; y solo entonces, si `decide`, clasificar.
 */
namespace hyperion::gpu_loop {

struct GpuSnapshot {
    double util_pct = 0;
    double mem_util_pct = 0;
    double power_mw = 0;
    double sm_clock_mhz = 0;
    double temperature_c = 0;
};

struct TrackerConfig {
    double activity_threshold_pct = 5.0;
    int64_t min_active_ns = 0;
};

struct TrackerStep {
    bool active_start = false;  // primera muestra de una actividad nueva
    bool decide = false;        // hay que clasificar ahora (una vez por fase)
    bool ended = false;         // la actividad terminó en esta muestra
};

class GpuActivityTracker {
public:
    explicit GpuActivityTracker(TrackerConfig cfg) : cfg_(cfg) {}

    TrackerStep step(const GpuSnapshot& s, int64_t now_ns) {
        TrackerStep r;
        const bool active_now = s.util_pct > cfg_.activity_threshold_pct;
        if (active_now && !is_active_) {
            is_active_ = true;
            active_since_ns_ = now_ns;
            decided_ = false;
            r.active_start = true;
        }
        if (active_now && !decided_ && now_ns - active_since_ns_ >= cfg_.min_active_ns) {
            decided_ = true;
            r.decide = true;
        } else if (!active_now && is_active_) {
            is_active_ = false;
            r.ended = true;
        }
        return r;
    }

    bool is_active() const { return is_active_; }
    /** Instante (ns) en que empezo la actividad actual; sirve para atribuir una decision a SU fase. */
    int64_t active_since_ns() const { return active_since_ns_; }

private:
    TrackerConfig cfg_;
    bool is_active_ = false;
    bool decided_ = false;
    int64_t active_since_ns_ = 0;
};

}  // namespace hyperion::gpu_loop

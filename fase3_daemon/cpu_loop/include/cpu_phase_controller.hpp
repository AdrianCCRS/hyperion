#pragma once
#include <cstdint>
#include <functional>

/**
 * @file
 * @brief Máquina de decisión del loop de CPU del daemon (§4.1/§4.3 puntos
 * 2 y 5 del plan de realineación).
 *
 * Mismo principio de diseño que common/telemetry/include/telemetry/
 * gpu_clock_controller.hpp: esta clase NO clasifica nada (recibe la
 * etiqueta ya decidida por la inferencia del modelo de Fase 2, corrida
 * aguas arriba en el tick de ~1ms) y NO escribe frecuencia por su cuenta
 * (delega en un `FrequencySetter` inyectado, para poder probarse sin
 * hardware real -- mismo patrón que `orchestrator/campaign.py` inyecta
 * `apply_frequency()`). Solo decide SI vale la pena actuar en este tick.
 *
 * Diferencia deliberada frente a GpuClockController: no hay `min_dwell_ns`
 * aquí. El plan (§4.3 punto 5) es explícito: "solo si la clase cambió
 * respecto al tick anterior, invocar la actuación de frecuencia" -- sin
 * piso de permanencia mínima, porque escribir scaling_min_freq/max_freq en
 * CPU es órdenes de magnitud más barato que bloquear el reloj de GPU (ver
 * el comentario de archivo de gpu_clock_controller.hpp para el contraste).
 *
 * Señal de coordinación CPU-GPU (§4.1, "Señal de coordinación"): mientras
 * el loop de GPU reporte actividad, el loop de CPU debe forzar el piso de
 * frecuencia sin importar lo que diga el clasificador ese ciclo -- porque
 * un cudaDeviceSynchronize() bloqueante (forzado por el shim de
 * fase3_daemon/shim/) hace que la CPU aparezca "ocupada" cuando en
 * realidad solo espera a la GPU. `on_window()` recibe ese flag como
 * parámetro explícito -- esta clase no lee la variable atómica compartida
 * por sí sola, el llamador se la pasa ya resuelta (mismo principio de
 * "esta clase no descubre nada por su cuenta" que GpuClockController).
 */
namespace hyperion::cpu_loop {

    enum class CpuPhaseLabel : uint8_t { ComputeBound, MemoryBound };

    /** Política de una clase (§3.4/§3.5): o bien un nivel de frecuencia
     * objetivo, o "no actuar" (chosen_khz se ignora si actuar==false). */
    struct CpuPolicyEntry {
        bool actuar = false;
        unsigned int target_freq_khz = 0;
    };

    struct CpuPhaseControllerConfig {
        CpuPolicyEntry compute_bound;
        CpuPolicyEntry memory_bound;
        /** Piso de frecuencia a forzar mientras gpu_active esté activo en
         * on_window() (§4.1, "Señal de coordinación") -- 0 = no hay piso
         * configurado, gpu_active se ignora. */
        unsigned int gpu_active_floor_khz = 0;
    };

    /** Resultado de un tick, para logging/CSV (§4.3 punto 10: "features
     * leídas, clase inferida, frecuencia aplicada, tiempo de inferencia,
     * tiempo de actuación" -- los dos últimos los mide el llamador
     * alrededor de la inferencia/de esta llamada, no esta clase). */
    struct CpuWindowDecision {
        CpuPhaseLabel label;
        bool gpu_active_override;      // true si este tick se forzó el piso por señal de coordinación
        bool actuation_attempted;      // true si se llamó al FrequencySetter este tick
        bool actuation_failed;         // true si actuation_attempted y el setter devolvió false
        unsigned int target_freq_khz;  // 0 si la política de la clase es "no actuar" y no hay override GPU
    };

    class CpuPhaseController {
    public:
        using FrequencySetter = std::function<bool(unsigned int khz)>;

        CpuPhaseController(CpuPhaseControllerConfig config, FrequencySetter set_frequency)
            : config_(config), set_frequency_(std::move(set_frequency)) {}

        /**
         * @brief Llamar una vez por ventana (~1ms), con la clase ya
         * inferida por el modelo de Fase 2 para esa ventana.
         *
         * @param label compute_bound/memory_bound de esta ventana.
         * @param gpu_active true si el loop de GPU reporta actividad ahora
         *   mismo -- fuerza gpu_active_floor_khz sin importar `label`.
         */
        CpuWindowDecision on_window(CpuPhaseLabel label, bool gpu_active) {
            CpuWindowDecision decision{};
            decision.label = label;
            decision.gpu_active_override = gpu_active && config_.gpu_active_floor_khz != 0;

            unsigned int desired_khz;
            bool policy_actuar;
            if (decision.gpu_active_override) {
                desired_khz = config_.gpu_active_floor_khz;
                policy_actuar = true;
            } else {
                const CpuPolicyEntry& policy =
                    (label == CpuPhaseLabel::ComputeBound) ? config_.compute_bound : config_.memory_bound;
                desired_khz = policy.target_freq_khz;
                policy_actuar = policy.actuar;
            }
            decision.target_freq_khz = policy_actuar ? desired_khz : 0;

            if (!policy_actuar) {
                // Política "no actuar" para esta clase: nunca se llama al
                // setter, sin importar si la clase cambió -- no hay nada
                // que aplicar.
                current_label_ = label;
                has_decided_once_ = true;
                return decision;
            }

            // Actúa solo si: es la primera decisión, la clase efectiva
            // cambió desde el último tick, o el override de GPU acaba de
            // activarse/desactivarse (ambos casos cuentan como "cambio").
            const bool effective_label_changed =
                !has_decided_once_
                || decision.gpu_active_override != last_gpu_active_override_
                || (!decision.gpu_active_override && label != current_label_);

            if (!effective_label_changed) {
                decision.actuation_attempted = false;
                current_label_ = label;
                has_decided_once_ = true;
                return decision;
            }

            const bool ok = set_frequency_(desired_khz);
            decision.actuation_attempted = true;
            decision.actuation_failed = !ok;
            current_label_ = label;
            last_gpu_active_override_ = decision.gpu_active_override;
            has_decided_once_ = true;
            return decision;
        }

        CpuPhaseLabel current_label() const noexcept { return current_label_; }
        bool has_decided_once() const noexcept { return has_decided_once_; }

    private:
        CpuPhaseControllerConfig config_;
        FrequencySetter set_frequency_;
        CpuPhaseLabel current_label_ = CpuPhaseLabel::ComputeBound;
        bool last_gpu_active_override_ = false;
        bool has_decided_once_ = false;
    };

}  // namespace hyperion::cpu_loop

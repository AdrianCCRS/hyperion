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
 * el loop de GPU reporte actividad, el loop de CPU nunca debe pedir una
 * frecuencia POR DEBAJO de `gpu_active_floor_khz`. Es una barrera, no un
 * objetivo: el piso se usa para elevar una petición demasiado baja, nunca
 * para bajar la frecuencia por su cuenta.
 *
 * Esta semántica corrige la que describe §4.1 del plan de realineación
 * ("mientras gpu_util_pct reporte actividad, forzar el reloj de CPU al
 * mínimo"), cuyo supuesto explícito --- "si la CPU está de verdad
 * bloqueada esperando, bajar su reloj casi no afecta el consumo" ---
 * quedó REFUTADO por medición (F1-XDEV-006, 2026-09-13): fijar la CPU al
 * mínimo durante una carga GPU la alarga 63-66% (134/134 y 1632/1632
 * pares más lentos en dos campañas), con la fase en régimen estacionario
 * casi duplicada (+93.8%) y una degradación que crece con el tamaño del
 * problema. La medida "defensiva" original, aplicada tal cual, degradaría
 * el tiempo en vez de ahorrar energía. El shim de blocking-sync
 * (common/hpc/native/blocking_sync_shim.cpp) sigue siendo la parte válida
 * del mecanismo: evita que una espera bloqueante se vea como IPC alto
 * ante el clasificador. Lo que se retira es bajar el reloj por esa señal.
 *
 * `on_window()` recibe el flag de actividad de GPU como parámetro
 * explícito -- esta clase no lee la variable atómica compartida por sí
 * sola, el llamador se la pasa ya resuelta (mismo principio de "esta
 * clase no descubre nada por su cuenta" que GpuClockController).
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
        /** Frecuencia MÍNIMA que el loop puede pedir mientras gpu_active
         * esté activo en on_window(): una petición de la política por
         * debajo de este valor se eleva hasta él. Nunca baja la frecuencia
         * por sí solo, y si la política de la clase es "no actuar" no se
         * escribe nada (ver la nota sobre F1-XDEV-006 en la cabecera del
         * archivo). 0 = sin barrera, gpu_active se ignora. */
        unsigned int gpu_active_floor_khz = 0;
    };

    /** Resultado de un tick, para logging/CSV (§4.3 punto 10: "features
     * leídas, clase inferida, frecuencia aplicada, tiempo de inferencia,
     * tiempo de actuación" -- los dos últimos los mide el llamador
     * alrededor de la inferencia/de esta llamada, no esta clase). */
    struct CpuWindowDecision {
        CpuPhaseLabel label;
        bool gpu_floor_clamped;        // true si la barrera de GPU elevó la petición de la política este tick
        bool actuation_attempted;      // true si se llamó al FrequencySetter este tick
        bool actuation_failed;         // true si actuation_attempted y el setter devolvió false
        unsigned int target_freq_khz;  // 0 si la política de la clase es "no actuar" (nunca se escribe nada)
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
            current_label_ = label;
            has_decided_once_ = true;

            const CpuPolicyEntry& policy =
                (label == CpuPhaseLabel::ComputeBound) ? config_.compute_bound : config_.memory_bound;

            if (!policy.actuar) {
                // Política "no actuar" para esta clase: no se escribe nada,
                // ni siquiera con la GPU activa. La barrera de GPU solo
                // puede ELEVAR una petición que la política ya decidió
                // hacer; nunca introduce una escritura por su cuenta,
                // porque bajar el reloj durante carga GPU degrada el
                // tiempo (F1-XDEV-006, ver cabecera del archivo).
                return decision;
            }

            unsigned int desired_khz = policy.target_freq_khz;
            if (gpu_active && config_.gpu_active_floor_khz > desired_khz) {
                desired_khz = config_.gpu_active_floor_khz;
                decision.gpu_floor_clamped = true;
            }
            decision.target_freq_khz = desired_khz;

            // Actúa solo si la frecuencia pedida cambia respecto de la
            // última efectivamente aplicada. Esto cubre de una sola vez el
            // cambio de clase, el enganche/desenganche de la barrera de
            // GPU y el arranque en frío, y evita reescribir el mismo valor
            // cuando dos clases comparten nivel.
            if (last_applied_khz_ == desired_khz) {
                return decision;
            }

            const bool ok = set_frequency_(desired_khz);
            decision.actuation_attempted = true;
            decision.actuation_failed = !ok;
            if (ok) {
                last_applied_khz_ = desired_khz;
            }
            return decision;
        }

        CpuPhaseLabel current_label() const noexcept { return current_label_; }
        bool has_decided_once() const noexcept { return has_decided_once_; }
        /** Última frecuencia efectivamente aplicada (0 = ninguna todavía). */
        unsigned int last_applied_khz() const noexcept { return last_applied_khz_; }

    private:
        CpuPhaseControllerConfig config_;
        FrequencySetter set_frequency_;
        CpuPhaseLabel current_label_ = CpuPhaseLabel::ComputeBound;
        unsigned int last_applied_khz_ = 0;
        bool has_decided_once_ = false;
    };

}  // namespace hyperion::cpu_loop

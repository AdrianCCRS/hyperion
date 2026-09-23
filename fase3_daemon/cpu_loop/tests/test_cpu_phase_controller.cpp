#include "cpu_phase_controller.hpp"

#include <vector>

int main() {
    using namespace hyperion::cpu_loop;

    // La barrera de GPU es un piso ALTO deliberadamente: mientras la GPU
    // trabaja, el loop de CPU no puede pedir menos que esto (F1-XDEV-006:
    // bajar la CPU durante carga GPU la alarga 63-66%).
    CpuPhaseControllerConfig config{};
    config.compute_bound = {true, 3600000};   // actuar, F0-equivalente
    config.memory_bound = {true, 800000};     // actuar, F8-equivalente
    config.gpu_active_floor_khz = 3200000;    // barrera: nunca por debajo de 3.2 GHz con GPU activa

    std::vector<unsigned int> applied;
    auto setter = [&applied](unsigned int khz) { applied.push_back(khz); return true; };
    CpuPhaseController controller(config, setter);

    // Primer tick: siempre actúa (no hay frecuencia aplicada todavía).
    auto d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 1;
    if (d.target_freq_khz != 3600000) return 2;
    if (applied.size() != 1 || applied[0] != 3600000) return 3;

    // Misma clase de nuevo: no debe actuar (a diferencia de GPU, aquí no
    // hay min_dwell -- pero SÍ hay "solo si la petición cambió").
    d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (d.actuation_attempted) return 4;
    if (applied.size() != 1) return 5;

    // Clase cambia -> debe actuar, sin GPU activa no hay barrera.
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 6;
    if (d.gpu_floor_clamped) return 7;
    if (d.target_freq_khz != 800000) return 8;
    if (applied.size() != 2 || applied[1] != 800000) return 9;

    // GPU se activa con la misma clase memory_bound: la política pediría
    // 800 MHz, la barrera lo ELEVA a 3.2 GHz. Esta es la corrección de
    // F1-XDEV-006: antes este caso bajaba el reloj, ahora lo protege.
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/true);
    if (!d.gpu_floor_clamped) return 10;
    if (!d.actuation_attempted) return 11;
    if (d.target_freq_khz != 3200000) return 12;
    if (applied.size() != 3 || applied[2] != 3200000) return 13;

    // GPU sigue activa, misma clase -> la petición no cambia, no reactúa.
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/true);
    if (d.actuation_attempted) return 14;
    if (applied.size() != 3) return 15;

    // GPU activa con compute_bound: la política ya pide 3.6 GHz, por
    // encima de la barrera -> la barrera no interviene y debe aplicarse
    // 3.6 GHz, no el piso.
    d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/true);
    if (d.gpu_floor_clamped) return 16;
    if (d.target_freq_khz != 3600000) return 17;
    if (!d.actuation_attempted) return 18;
    if (applied.size() != 4 || applied[3] != 3600000) return 19;

    // GPU se desactiva con memory_bound -> sin barrera, vuelve a 800 MHz.
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    if (d.gpu_floor_clamped) return 20;
    if (!d.actuation_attempted) return 21;
    if (d.target_freq_khz != 800000) return 22;
    if (applied.size() != 5 || applied[4] != 800000) return 23;

    // Política "no actuar" para una clase: nunca llama al setter, NI
    // SIQUIERA con la GPU activa. La barrera solo eleva una petición que
    // la política ya decidió hacer; nunca introduce una escritura propia
    // (ese era justo el mecanismo refutado por F1-XDEV-006).
    CpuPhaseControllerConfig no_actuar_config{};
    no_actuar_config.compute_bound = {false, 0};
    no_actuar_config.memory_bound = {false, 0};
    no_actuar_config.gpu_active_floor_khz = 3200000;
    std::vector<unsigned int> never_applied;
    CpuPhaseController passive(
        no_actuar_config,
        [&never_applied](unsigned int khz) { never_applied.push_back(khz); return true; });
    passive.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    passive.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    passive.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/true);
    passive.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/true);
    if (!never_applied.empty()) return 24;

    // Dos clases que comparten nivel: el cambio de clase no debe producir
    // una reescritura del mismo valor.
    CpuPhaseControllerConfig same_level{};
    same_level.compute_bound = {true, 2000000};
    same_level.memory_bound = {true, 2000000};
    std::vector<unsigned int> same_applied;
    CpuPhaseController flat(
        same_level, [&same_applied](unsigned int khz) { same_applied.push_back(khz); return true; });
    flat.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    flat.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    if (same_applied.size() != 1) return 25;

    // Setter que falla: se reporta el fallo y NO se registra como aplicada,
    // de modo que el siguiente tick reintenta en vez de creer que el
    // hardware ya está en ese nivel.
    std::vector<unsigned int> attempted;
    CpuPhaseController flaky(config, [&attempted](unsigned int khz) {
        attempted.push_back(khz);
        return false;
    });
    d = flaky.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 26;
    if (!d.actuation_failed) return 27;
    if (flaky.last_applied_khz() != 0) return 28;
    d = flaky.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 29;  // reintenta tras el fallo
    if (attempted.size() != 2) return 30;

    return 0;
}

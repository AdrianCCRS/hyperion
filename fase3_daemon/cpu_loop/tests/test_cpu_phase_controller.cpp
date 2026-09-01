#include "cpu_phase_controller.hpp"

#include <vector>

int main() {
    using namespace hyperion::cpu_loop;

    CpuPhaseControllerConfig config{};
    config.compute_bound = {true, 3600000};   // actuar, F0-equivalente
    config.memory_bound = {true, 800000};     // actuar, F4-equivalente
    config.gpu_active_floor_khz = 800000;

    std::vector<unsigned int> applied;
    auto setter = [&applied](unsigned int khz) { applied.push_back(khz); return true; };
    CpuPhaseController controller(config, setter);

    // Primer tick: siempre actúa.
    auto d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 1;
    if (d.target_freq_khz != 3600000) return 2;
    if (applied.size() != 1 || applied[0] != 3600000) return 3;

    // Misma clase de nuevo: no debe actuar (a diferencia de GPU, aquí no
    // hay min_dwell -- pero SÍ hay "solo si cambió", que es justo esto).
    d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (d.actuation_attempted) return 4;
    if (applied.size() != 1) return 5;

    // Clase cambia -> debe actuar.
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 6;
    if (d.target_freq_khz != 800000) return 7;
    if (applied.size() != 2 || applied[1] != 800000) return 8;

    // GPU se activa -> debe forzar el piso, sin importar que el
    // clasificador siga diciendo memory_bound (mismo valor, pero el
    // override cuenta como una transición y debe re-actuar).
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/true);
    if (!d.gpu_active_override) return 9;
    if (!d.actuation_attempted) return 10;
    if (d.target_freq_khz != 800000) return 11;
    if (applied.size() != 3) return 12;

    // GPU sigue activa, misma clase reportada -> no reactuar (ya está en
    // el piso).
    d = controller.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/true);
    if (d.actuation_attempted) return 13;
    if (applied.size() != 3) return 14;

    // GPU se activa mientras el clasificador dice compute_bound -> el
    // piso debe ganar, no 3600000.
    d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/true);
    if (d.target_freq_khz != 800000) return 15;
    if (d.actuation_attempted) return 16;  // ya estaba en el piso desde el tick anterior

    // GPU se desactiva, clasificador vuelve a compute_bound -> debe
    // re-actuar hacia 3600000 (transición fuera del override).
    d = controller.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (d.gpu_active_override) return 17;
    if (!d.actuation_attempted) return 18;
    if (d.target_freq_khz != 3600000) return 19;
    if (applied.size() != 4 || applied[3] != 3600000) return 20;

    // Política "no actuar" para una clase: nunca llama al setter.
    CpuPhaseControllerConfig no_actuar_config{};
    no_actuar_config.compute_bound = {false, 0};
    no_actuar_config.memory_bound = {false, 0};
    std::vector<unsigned int> never_applied;
    CpuPhaseController passive(
        no_actuar_config,
        [&never_applied](unsigned int khz) { never_applied.push_back(khz); return true; });
    passive.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    passive.on_window(CpuPhaseLabel::MemoryBound, /*gpu_active=*/false);
    if (!never_applied.empty()) return 21;

    // Setter que falla no debe hacer crecer applied ni ocultarse.
    CpuPhaseController flaky(config, [](unsigned int) { return false; });
    d = flaky.on_window(CpuPhaseLabel::ComputeBound, /*gpu_active=*/false);
    if (!d.actuation_attempted) return 22;
    if (!d.actuation_failed) return 23;

    return 0;
}

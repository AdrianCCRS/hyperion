// Puerto de los tests de fase3_daemon/tests/test_activity_poller.py (mismos casos).
#include <vector>

#include "check.hpp"
#include "gpu_activity_tracker.hpp"

using namespace hyperion::gpu_loop;

static GpuSnapshot util(double u) { GpuSnapshot s; s.util_pct = u; return s; }
constexpr int64_t S = 1'000'000'000;

// Sin ventana sostenida: una decision por cada transicion idle->activo, no una por muestra activa.
static void test_flanco_de_subida_una_decision_por_fase() {
    GpuActivityTracker t({5.0, 0});
    std::vector<double> readings{1, 2, 50, 60, 55, 1, 80};
    std::vector<int> decide_at, end_at;
    for (size_t i = 0; i < readings.size(); ++i) {
        auto r = t.step(util(readings[i]), static_cast<int64_t>(i));
        if (r.decide) decide_at.push_back(static_cast<int>(i));
        if (r.ended) end_at.push_back(static_cast<int>(i));
    }
    CHECK((decide_at == std::vector<int>{2, 6}));
    CHECK((end_at == std::vector<int>{5}));
}

// Con min_active=3 s y muestras cada 1 s: activo desde t=1 s, la decision sale en t=4 s.
static void test_ventana_sostenida_decide_tras_min_active() {
    GpuActivityTracker t({5.0, 3 * S});
    int64_t decided_at = -1;
    std::vector<double> readings{1, 50, 60, 70, 80, 90};
    for (size_t i = 0; i < readings.size(); ++i) {
        auto r = t.step(util(readings[i]), i * S);
        if (r.decide) decided_at = i * S;
    }
    CHECK(decided_at == 4 * S);
}

// Un pico corto (2 s < 3 s) no genera decision, pero si el fin de la actividad.
static void test_pico_corto_no_decide_pero_termina() {
    GpuActivityTracker t({5.0, 3 * S});
    bool decided = false, ended = false;
    int64_t end_t = -1;
    std::vector<double> readings{1, 50, 60, 1, 1};
    for (size_t i = 0; i < readings.size(); ++i) {
        auto r = t.step(util(readings[i]), i * S);
        decided |= r.decide;
        if (r.ended) { ended = true; end_t = i * S; }
    }
    CHECK(!decided);
    CHECK(ended && end_t == 3 * S);
}

// Un evento por fase y active_start al inicio de cada actividad.
static void test_un_decide_por_fase_y_active_start_por_actividad() {
    GpuActivityTracker t({5.0, 2 * S});
    std::vector<double> readings{50, 60, 70, 1, 80, 90, 95};
    std::vector<int64_t> starts, decides;
    for (size_t i = 0; i < readings.size(); ++i) {
        auto r = t.step(util(readings[i]), i * S);
        if (r.active_start) starts.push_back(i * S);
        if (r.decide) decides.push_back(i * S);
    }
    CHECK((starts == std::vector<int64_t>{0, 4 * S}));
    CHECK((decides == std::vector<int64_t>{2 * S, 6 * S}));
}

int main() {
    test_flanco_de_subida_una_decision_por_fase();
    test_ventana_sostenida_decide_tras_min_active();
    test_pico_corto_no_decide_pero_termina();
    test_un_decide_por_fase_y_active_start_por_actividad();
    std::printf("gpu_activity_tracker_test: OK\n");
}

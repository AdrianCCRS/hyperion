// Puerto de los tests de fase3_daemon/tests/test_gpu_classifier.py (construccion del vector).
#include <cmath>
#include <string>
#include <vector>

#include "check.hpp"
#include "gpu_window_classifier.hpp"

using namespace hyperion::gpu_loop;

static GpuSnapshot snap(double util, double mem, double power = 0, double clock = 0) {
    GpuSnapshot s; s.util_pct = util; s.mem_util_pct = mem; s.power_mw = power; s.sm_clock_mhz = clock; return s;
}

static void test_arranque_en_frio_usa_la_instantanea() {
    GpuFeatureWindow w({"gpu_util_pct_median", "gpu_mem_util_pct_median", "gpu_mem_util_pct_std"}, 10);
    auto v = w.build(snap(42, 33));
    CHECK((v == std::vector<float>{42.f, 33.f, 0.f}));  // mediana de 1 = esa muestra, std de 1 = 0
}

static void test_mediana_y_std_muestral_sobre_la_ventana() {
    GpuFeatureWindow w({"gpu_mem_util_pct_median", "gpu_mem_util_pct_std"}, 10);
    for (double m : {10.0, 20.0, 30.0}) w.record(snap(50, m));
    auto v = w.build(snap(50, 30));
    CHECK(v[0] == 20.f);                        // mediana de [10, 20, 30]
    CHECK(std::fabs(v[1] - 10.f) < 1e-5f);      // std muestral (ddof=1) de [10, 20, 30] = 10
}

static void test_mediana_de_cantidad_par_es_la_media_de_los_centrales() {
    GpuFeatureWindow w({"gpu_util_pct_median"}, 10);
    for (double u : {10.0, 20.0, 30.0, 100.0}) w.record(snap(u, 0));
    CHECK(w.build(snap(0, 0))[0] == 25.f);      // (20 + 30) / 2, como statistics.median
}

static void test_la_ventana_descarta_lo_mas_antiguo() {
    GpuFeatureWindow w({"gpu_util_pct_median"}, 3);
    for (double u : {1.0, 2.0, 100.0, 100.0, 100.0}) w.record(snap(u, 0));
    CHECK(w.size() == 3);
    CHECK(w.build(snap(0, 0))[0] == 100.f);
}

static void test_reset_vacia_el_hueco_ocioso_previo() {
    GpuFeatureWindow w({"gpu_util_pct_median"}, 50);
    for (int i = 0; i < 10; ++i) w.record(snap(0, 0));   // hueco ocioso
    w.reset();
    for (double u : {90.0, 100.0, 95.0}) w.record(snap(u, 0));
    CHECK(w.build(snap(95, 0))[0] == 95.f);              // sin los ceros del hueco
}

static void test_nombre_desconocido_falla_al_construir() {
    bool threw = false;
    try { GpuFeatureWindow w({"gpu_util_pct_median", "algo_no_entrenado"}, 5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { GpuFeatureWindow w({"gpu_temperatura_median"}, 5); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_orden_de_las_variables_es_el_del_modelo() {
    GpuFeatureWindow w({"gpu_mem_util_pct_std", "gpu_util_pct_median"}, 10);
    w.record(snap(80, 10)); w.record(snap(80, 30));
    auto v = w.build(snap(80, 30));
    CHECK(std::fabs(v[0] - 14.142135f) < 1e-4f && v[1] == 80.f);  // std primero, tal como pide el modelo
}

int main() {
    test_arranque_en_frio_usa_la_instantanea();
    test_mediana_y_std_muestral_sobre_la_ventana();
    test_mediana_de_cantidad_par_es_la_media_de_los_centrales();
    test_la_ventana_descarta_lo_mas_antiguo();
    test_reset_vacia_el_hueco_ocioso_previo();
    test_nombre_desconocido_falla_al_construir();
    test_orden_de_las_variables_es_el_del_modelo();
    std::printf("gpu_window_classifier_test: OK\n");
}

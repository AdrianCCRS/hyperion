#include "check.hpp"
#include "gpu_decision.hpp"

using namespace hyperion::gpu_loop;

static void test_confianza_alta_decide_la_clase() {
    CHECK(decide(0.95f, 0.70f) == GpuDecision::kMemoryBound);
    CHECK(decide(0.05f, 0.70f) == GpuDecision::kComputeBound);  // P(memory)=0.05 -> confianza 0.95 en compute
}

static void test_confianza_baja_es_revisar_en_ambos_lados() {
    CHECK(decide(0.60f, 0.70f) == GpuDecision::kRevisar);
    CHECK(decide(0.40f, 0.70f) == GpuDecision::kRevisar);
    CHECK(decide(0.50f, 0.70f) == GpuDecision::kRevisar);
}

static void test_el_umbral_es_inclusivo() {
    CHECK(decide(0.75f, 0.75f) == GpuDecision::kMemoryBound);
    CHECK(decide(0.25f, 0.75f) == GpuDecision::kComputeBound);
}

// Umbral 0.5 = sin abstencion: reproduce el comportamiento anterior (P > 0.5), salvo el empate exacto.
static void test_umbral_medio_equivale_a_no_abstenerse() {
    CHECK(decide(0.51f, 0.5f) == GpuDecision::kMemoryBound);
    CHECK(decide(0.49f, 0.5f) == GpuDecision::kComputeBound);
}

int main() {
    test_confianza_alta_decide_la_clase();
    test_confianza_baja_es_revisar_en_ambos_lados();
    test_el_umbral_es_inclusivo();
    test_umbral_medio_equivale_a_no_abstenerse();
    std::printf("gpu_decision_test: OK\n");
}

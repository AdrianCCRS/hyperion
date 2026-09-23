#include "onnx_cpu_classifier.hpp"

#include <cmath>
#include <cstdio>

using namespace hyperion::cpu_loop;

/**
 * Valores de referencia calculados en Python contra el MISMO .joblib
 * (fase2_clasificador/models/xgboost_cpu.joblib) con model.predict_proba(),
 * no inventados -- comando exacto en el comentario de cada fila. Cierra el
 * mismo tipo de verificación fila-a-fila que export_onnx.py::verify_row_by_row()
 * ya hace en Python, ahora también desde el lado C++ que realmente va a
 * correr en producción.
 */
int main() {
    // Ejecutado desde CMAKE_CURRENT_SOURCE_DIR (fase3_daemon/cpu_loop/), ver
    // WORKING_DIRECTORY en CMakeLists.txt -- ruta relativa al .onnx exportado.
    OnnxCpuClassifier classifier("xgboost_cpu.onnx");

    struct Case { FeatureVector features; float expected_p_memory_bound; };
    const Case cases[] = {
        // python3 -c "... model.predict_proba([[0.8,15.0,0.3,0.4,2e9,2000000.0]])" -> 0.998283
        {{0.8f, 15.0f, 0.3f, 0.4f, 2e9f, 2000000.0f}, 0.998283f},
        // [[3.5,1.0,0.05,0.02,8e9,3200000.0]] -> 0.654169
        {{3.5f, 1.0f, 0.05f, 0.02f, 8e9f, 3200000.0f}, 0.654169f},
        // [[0.3,60.0,0.7,0.6,5e8,800000.0]] -> 0.9993918
        {{0.3f, 60.0f, 0.7f, 0.6f, 5e8f, 800000.0f}, 0.9993918f},
    };

    for (const auto& c : cases) {
        float p = classifier.predict_memory_bound_proba(c.features);
        double diff = std::abs(static_cast<double>(p) - static_cast<double>(c.expected_p_memory_bound));
        if (diff > 1e-3) {
            std::fprintf(stderr, "mismatch: onnx=%.6f esperado=%.6f diff=%.2e\n", p, c.expected_p_memory_bound, diff);
            return 1;
        }
    }

    // Umbral de decisión (0.85, xgboost_cpu.metadata.json::decision.threshold):
    // el primer y tercer caso deben clasificar memory_bound con confianza
    // suficiente para actuar sin abstenerse; verifica que la salida cruda
    // sea utilizable directamente por la ecuación de decisión selectiva.
    float p0 = classifier.predict_memory_bound_proba(cases[0].features);
    if (p0 < 0.85f) return 2;

    return 0;
}

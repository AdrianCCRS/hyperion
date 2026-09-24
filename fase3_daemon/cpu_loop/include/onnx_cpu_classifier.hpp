#pragma once
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "cpu_feature_builder.hpp"

/**
 * @file
 * @brief Envoltorio de ONNX Runtime C++ para el candidato CPU exportado por
 * `fase3_daemon/cpu_loop/export_onnx.py` (§4.3 punto 3 del plan).
 *
 * El grafo se exportó con `zipmap=False` (ver export_onnx.py), así que
 * `session.Run()` devuelve DOS tensores de salida, en este orden fijo
 * (verificado contra el mismo grafo en Python, `verify_row_by_row()`):
 *   [0] label:       int64,  forma [N]     -- 0/1, la clase predicha por XGBoost
 *   [1] probabilities: float32, forma [N,2] -- [P(compute_bound), P(memory_bound)]
 *
 * Esta clase NUNCA decide el umbral de confianza (0.85, §metodologia-lofo-cpu)
 * -- solo expone P(memory_bound); la decisión selectiva vive donde ya vivía
 * en Python (ecuación de decisión selectiva), para no duplicar esa lógica en
 * dos lenguajes.
 */
namespace hyperion::cpu_loop {

class OnnxCpuClassifier {
public:
    /** Carga el modelo desde `model_path`. Lanza `std::runtime_error` si el
     * grafo no tiene exactamente la forma de entrada/salida esperada -- un
     * modelo con una firma distinta debe fallar en la carga, no en el
     * primer tick de inferencia con un error de ORT poco claro. */
    explicit OnnxCpuClassifier(const std::string& model_path, int intra_op_threads = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "hyperion_cpu_loop"),
          session_(env_, model_path.c_str(), make_options(intra_op_threads)) {
        if (session_.GetInputCount() != 1) {
            throw std::runtime_error("OnnxCpuClassifier: se esperaba exactamente 1 entrada, el grafo tiene "
                                     + std::to_string(session_.GetInputCount()));
        }
        if (session_.GetOutputCount() != 2) {
            throw std::runtime_error("OnnxCpuClassifier: se esperaba exactamente 2 salidas (label, probabilities; "
                                     "zipmap=False), el grafo tiene " + std::to_string(session_.GetOutputCount()));
        }
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_names_[0] = session_.GetOutputNameAllocated(0, allocator).get();
        output_names_[1] = session_.GetOutputNameAllocated(1, allocator).get();
    }

    /** Probabilidad de memory_bound para un único vector de 6 variables
     * (una fila, nunca en lote -- mismo patrón de medición que
     * `export_gpu_historical_candidate.py::measure_latency`, fila a fila,
     * porque así decide el loop real, un tick a la vez). */
    float predict_memory_bound_proba(const FeatureVector& features) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(features.size())};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(features.data()), features.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_names_[0].c_str(), output_names_[1].c_str()};

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                     output_names, 2);
        // outputs[1] es "probabilities", forma [1,2]: [P(False), P(True)] --
        // el modelo se entreno con y = phase_label_train.eq("memory_bound"),
        // asi que el indice 1 (True) es P(memory_bound). Ver el docstring
        // de la clase y gpu_loop/classifier.py::classify() para el mismo
        // contrato booleano del lado GPU.
        const float* proba = outputs[1].GetTensorData<float>();
        return proba[1];
    }

private:
    /** intra_op_threads > 0: pool acotado y sin espera activa (por defecto ORT crea un hilo por core que gira
     * entre inferencias); 0 = valores por defecto de ORT. */
    static Ort::SessionOptions make_options(int intra_op_threads) {
        Ort::SessionOptions o;
        if (intra_op_threads > 0) {
            o.SetIntraOpNumThreads(intra_op_threads);
            o.AddConfigEntry("session.intra_op.allow_spinning", "0");
        }
        return o;
    }
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;
    std::array<std::string, 2> output_names_;
};

}  // namespace hyperion::cpu_loop

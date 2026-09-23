#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

/**
 * @file
 * @brief Envoltorio de ONNX Runtime C++ para el random forest de GPU
 * (`export_onnx.py` de este directorio). Mismo patrón que
 * `cpu_loop/include/onnx_cpu_classifier.hpp`, pero con vector de variables de
 * largo variable (el candidato de GPU tiene 3 variables, el de CPU 6).
 *
 * El grafo se exporta con `zipmap=False`: dos salidas en orden fijo,
 *   [0] label int64 [N], [1] probabilities float32 [N,2] = [P(compute), P(memory)]
 * (el modelo se entrenó con y = etiqueta.eq("memory_bound"), así que el índice
 * 1 es memory_bound). Verificado fila a fila contra el .joblib al exportar.
 * Esta clase solo expone P(memory_bound); la clase = P > 0.5 (mismo criterio
 * que `RandomForestClassifier.predict`, que toma el argmax del promedio).
 */
namespace hyperion::gpu_loop {

inline std::vector<std::string> read_feature_names(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("no se pudo abrir el archivo de variables: " + path);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) throw std::runtime_error("archivo de variables vacío: " + path);
    return names;
}

class OnnxGpuClassifier {
public:
    OnnxGpuClassifier(const std::string& model_path, size_t feature_count)
        : env_(ORT_LOGGING_LEVEL_WARNING, "hyperion_gpu_loop"),
          session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr}),
          feature_count_(feature_count) {
        if (session_.GetInputCount() != 1)
            throw std::runtime_error("OnnxGpuClassifier: se esperaba 1 entrada, hay " + std::to_string(session_.GetInputCount()));
        if (session_.GetOutputCount() != 2)
            throw std::runtime_error("OnnxGpuClassifier: se esperaban 2 salidas (label, probabilities; zipmap=False), hay "
                                     + std::to_string(session_.GetOutputCount()));
        const auto in_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() != 2 || (in_shape[1] > 0 && static_cast<size_t>(in_shape[1]) != feature_count))
            throw std::runtime_error("OnnxGpuClassifier: el modelo espera otro número de variables que el archivo de variables");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_names_[0] = session_.GetOutputNameAllocated(0, allocator).get();
        output_names_[1] = session_.GetOutputNameAllocated(1, allocator).get();
    }

    /** P(memory_bound) para una fila (nunca en lote: así decide el loop real). */
    float predict_memory_bound_proba(std::vector<float>& features) {
        if (features.size() != feature_count_)
            throw std::invalid_argument("OnnxGpuClassifier: vector de tamaño inesperado");
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> shape{1, static_cast<int64_t>(features.size())};
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, features.data(), features.size(), shape.data(), shape.size());
        const char* in_names[] = {input_name_.c_str()};
        const char* out_names[] = {output_names_[0].c_str(), output_names_[1].c_str()};
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 2);
        return outputs[1].GetTensorData<float>()[1];
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    size_t feature_count_;
    std::string input_name_;
    std::array<std::string, 2> output_names_;
};

}  // namespace hyperion::gpu_loop

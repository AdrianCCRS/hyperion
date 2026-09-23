#pragma once
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file
 * @brief Espejo en C++ de `fase3_daemon/decision_log.py` (Plan_Fase3_Daemon.md
 * §0.1, requisito 2): mismo esquema JSONL, un renglón por decisión (un tick
 * de CPU aquí, una fase de GPU en el lado Python). Cualquier campo que se
 * agregue en un lado debe agregarse en el otro -- el análisis de Fase 4
 * necesita poder unir ambas fuentes con un solo parser.
 *
 * Sin dependencia de una biblioteca JSON externa: el esquema es plano (sin
 * anidamiento salvo `features`, un mapa string->float de tamaño fijo
 * conocido en compilación), así que se serializa a mano. No vale la pena
 * una dependencia nueva solo para esto.
 */
namespace hyperion::cpu_loop {

struct DecisionRecordField {
    std::string name;
    double value;
};

struct DecisionRecord {
    int64_t ts_ns = 0;
    std::string arm;     // "base" | "sombra" | "activo"
    std::string device = "cpu";
    std::optional<std::string> label;  // nullopt si no se llegó a clasificar
    std::optional<float> confidence;
    std::vector<DecisionRecordField> features;
    std::string policy_action = "n/a";  // "actuar" | "no_actuar" | "n/a"
    unsigned int target_freq_khz = 0;
    unsigned int applied_freq_khz = 0;
    bool written = false;
    bool write_failed = false;
    std::optional<int64_t> inference_time_ns;
    std::optional<int64_t> actuation_time_ns;
    std::optional<std::string> error;  // p.ej. FeatureBuildError, solo si aplica
};

namespace detail {

inline void json_escape_into(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            default: os << c;
        }
    }
    os << '"';
}

template <typename T>
inline void write_opt(std::ostringstream& os, const std::optional<T>& v) {
    if (v.has_value()) {
        os << *v;
    } else {
        os << "null";
    }
}

inline void write_opt_str(std::ostringstream& os, const std::optional<std::string>& v) {
    if (v.has_value()) {
        json_escape_into(os, *v);
    } else {
        os << "null";
    }
}

}  // namespace detail

inline std::string to_json(const DecisionRecord& r) {
    std::ostringstream os;
    os << "{\"actuation_time_ns\":";
    detail::write_opt(os, r.actuation_time_ns);
    os << ",\"applied_freq_khz\":" << r.applied_freq_khz;
    os << ",\"arm\":";
    detail::json_escape_into(os, r.arm);
    os << ",\"confidence\":";
    detail::write_opt(os, r.confidence);
    os << ",\"device\":";
    detail::json_escape_into(os, r.device);
    os << ",\"error\":";
    detail::write_opt_str(os, r.error);
    os << ",\"features\":{";
    for (size_t i = 0; i < r.features.size(); ++i) {
        if (i > 0) os << ',';
        detail::json_escape_into(os, r.features[i].name);
        os << ':' << r.features[i].value;
    }
    os << "}";
    os << ",\"inference_time_ns\":";
    detail::write_opt(os, r.inference_time_ns);
    os << ",\"label\":";
    detail::write_opt_str(os, r.label);
    os << ",\"policy_action\":";
    detail::json_escape_into(os, r.policy_action);
    os << ",\"target_freq_khz\":" << r.target_freq_khz;
    os << ",\"ts_ns\":" << r.ts_ns;
    os << ",\"write_failed\":" << (r.write_failed ? "true" : "false");
    os << ",\"written\":" << (r.written ? "true" : "false");
    os << "}";
    return os.str();
}

/**
 * @brief Append-only, una línea JSON por decisión, flush inmediato --
 * mismo criterio que `DecisionLogWriter` en decision_log.py (perder el
 * buffer de las últimas decisiones ante una salida sin manejar costaría
 * más que el costo de un flush por tick).
 */
class DecisionLogWriter {
public:
    explicit DecisionLogWriter(const std::string& path)
        : out_(path, std::ios::app) {}

    void write(const DecisionRecord& r) {
        out_ << to_json(r) << '\n';
        out_.flush();
    }

    bool is_open() const { return out_.is_open(); }

private:
    std::ofstream out_;
};

}  // namespace hyperion::cpu_loop

#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_activity_tracker.hpp"

/**
 * @file
 * @brief Buffer móvil de muestras NVML y construcción del vector de variables
 * del clasificador de GPU. Puerto de `fase3_daemon/gpu_loop/classifier.py`
 * (`HistoricalGpuClassifier._build_feature_vector`, `reset_window`).
 *
 * El modelo se entrenó con medianas/desviaciones de CORRIDAS completas; aquí
 * se aproximan con las muestras de la fase actual (la ventana se vacía al
 * inicio de cada actividad, ver GpuActivityTracker). Los nombres de variable
 * salen del archivo de features exportado junto al modelo -- nunca se
 * hardcodean; un nombre que este puerto no sabe derivar lanza al construir,
 * en vez de rellenar en silencio.
 *
 * Definiciones idénticas a Python: mediana = `statistics.median` (media de los
 * dos centrales si hay cantidad par); std = `statistics.stdev` (muestral,
 * ddof=1) y 0 con una sola muestra.
 */
namespace hyperion::gpu_loop {

namespace detail {

inline double field_of(const GpuSnapshot& s, const std::string& base) {
    if (base == "gpu_util_pct") return s.util_pct;
    if (base == "gpu_mem_util_pct") return s.mem_util_pct;
    if (base == "gpu_power_mw") return s.power_mw;
    if (base == "gpu_sm_clock_mhz") return s.sm_clock_mhz;
    throw std::invalid_argument("GpuWindowClassifier: campo desconocido: " + base);
}

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

}  // namespace detail

class GpuFeatureWindow {
public:
    GpuFeatureWindow(std::vector<std::string> feature_names, size_t capacity)
        : names_(std::move(feature_names)), capacity_(capacity) {
        for (const auto& n : names_) {  // falla al construir, no en el primer tick
            if (detail::ends_with(n, "_median")) detail::field_of(GpuSnapshot{}, n.substr(0, n.size() - 7));
            else if (detail::ends_with(n, "_std")) detail::field_of(GpuSnapshot{}, n.substr(0, n.size() - 4));
            else throw std::invalid_argument("GpuFeatureWindow no sabe derivar la columna del modelo: " + n);
        }
    }

    void reset() { window_.clear(); }

    void record(const GpuSnapshot& s) {
        if (window_.size() == capacity_) window_.pop_front();
        window_.push_back(s);
    }

    size_t size() const { return window_.size(); }
    const std::vector<std::string>& names() const { return names_; }

    /** Vector en el MISMO orden que `names()`. Con la ventana vacía usa la
     * instantánea `now` como única observación (mediana de 1 = esa muestra,
     * std de 1 = 0), el caso límite honesto, no una excepción especial. */
    std::vector<float> build(const GpuSnapshot& now) const {
        std::vector<GpuSnapshot> w(window_.begin(), window_.end());
        if (w.empty()) w.push_back(now);
        std::vector<float> row;
        row.reserve(names_.size());
        for (const auto& n : names_) {
            const bool is_median = detail::ends_with(n, "_median");
            const std::string base = n.substr(0, n.size() - (is_median ? 7 : 4));
            std::vector<double> v;
            v.reserve(w.size());
            for (const auto& s : w) v.push_back(detail::field_of(s, base));
            row.push_back(static_cast<float>(is_median ? median(v) : stdev(v)));
        }
        return row;
    }

private:
    static double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
    }
    static double stdev(const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double mean = 0;
        for (double x : v) mean += x;
        mean /= v.size();
        double ss = 0;
        for (double x : v) ss += (x - mean) * (x - mean);
        return std::sqrt(ss / (v.size() - 1));
    }

    std::vector<std::string> names_;
    size_t capacity_;
    std::deque<GpuSnapshot> window_;
};

}  // namespace hyperion::gpu_loop

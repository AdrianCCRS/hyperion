#pragma once
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

/**
 * @file
 * @brief Escritor de la señal de coordinación CPU-GPU (Bloque C, ítem C4): un
 * archivo de un byte ("1" activa / "0" inactiva) escrito ATÓMICAMENTE (temporal
 * + rename, nunca in situ) para que `cpu_loop_main` (que lo lee cada tick con
 * `gpu_active_reader.hpp`) nunca vea una escritura a medias. Puerto de
 * `fase3_daemon/gpu_loop/coordination.py::GpuActiveSignalWriter`.
 */
namespace hyperion::gpu_loop {

class GpuActiveWriter {
public:
    explicit GpuActiveWriter(std::string path) : path_(std::move(path)), tmp_(path_ + ".tmp") {
        const auto parent = std::filesystem::path(path_).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }

    /** true si la escritura y el rename tuvieron éxito. */
    bool write(bool active) {
        {
            std::ofstream out(tmp_, std::ios::binary | std::ios::trunc);
            if (!out) return false;
            out << (active ? '1' : '0');
            out.flush();
            if (!out) return false;
        }
        return std::rename(tmp_.c_str(), path_.c_str()) == 0;  // atómico en el mismo filesystem (POSIX)
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::string tmp_;
};

}  // namespace hyperion::gpu_loop

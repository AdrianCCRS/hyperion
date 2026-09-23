#pragma once
#include <fstream>
#include <string>

/**
 * @file
 * @brief Espejo en C++ de `fase3_daemon/gpu_loop/coordination.py` (Bloque
 * C, ítem C4) -- lee el mismo archivo de un byte que escribe el loop de
 * GPU (Python, proceso separado), atómicamente reemplazado en cada
 * transición de fase (`os.replace`, POSIX rename() atómico).
 *
 * Falla cerrado a `false` (GPU inactiva) ante CUALQUIER problema de
 * lectura: archivo ausente (el loop de GPU no ha arrancado todavía, o no
 * está corriendo en absoluto -- p.ej. brazo sin GPU), archivo vacío, o
 * contenido irreconocible. Es el mismo default que existía ANTES de que
 * esta señal existiera (`[]{ return false; }` en cpu_loop_main.cpp) --
 * nunca inventa actividad de GPU que no pudo confirmar. La asimetría de
 * riesgo es deliberada: un falso "inactiva" cuando la GPU sí está activa
 * solo cuesta la ganancia de la barrera de §0.2 (que la política de CPU
 * ya decidió no aplicar), nunca escribe una frecuencia que la política no
 * pidió; un falso "activa" no tiene ese mismo piso de seguridad.
 */
namespace hyperion::cpu_loop {

class GpuActiveFileReader {
public:
    explicit GpuActiveFileReader(std::string path) : path_(std::move(path)) {}

    bool read_active() const {
        std::ifstream in(path_, std::ios::binary);
        if (!in.is_open()) return false;
        char c = '\0';
        if (!in.get(c)) return false;
        return c == '1';
    }

private:
    std::string path_;
};

}  // namespace hyperion::cpu_loop

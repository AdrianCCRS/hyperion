#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "gpu_active_reader.hpp"

using namespace hyperion::cpu_loop;

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
}

// Archivo ausente -> false, el default seguro previo a que esta senal
// existiera (nunca fabrica actividad de GPU que no pudo confirmar).
static void test_archivo_ausente_es_false() {
    GpuActiveFileReader reader("/tmp/hyperion_test_gpu_active_no_existe.txt");
    std::remove("/tmp/hyperion_test_gpu_active_no_existe.txt");
    assert(reader.read_active() == false);
}

static void test_contenido_uno_es_true() {
    const std::string path = "/tmp/hyperion_test_gpu_active_uno.txt";
    write_file(path, "1");
    GpuActiveFileReader reader(path);
    assert(reader.read_active() == true);
    std::remove(path.c_str());
}

static void test_contenido_cero_es_false() {
    const std::string path = "/tmp/hyperion_test_gpu_active_cero.txt";
    write_file(path, "0");
    GpuActiveFileReader reader(path);
    assert(reader.read_active() == false);
    std::remove(path.c_str());
}

// Archivo vacio o con contenido irreconocible -> false, falla cerrado
// (nunca interpreta basura como "activa").
static void test_archivo_vacio_es_false() {
    const std::string path = "/tmp/hyperion_test_gpu_active_vacio.txt";
    write_file(path, "");
    GpuActiveFileReader reader(path);
    assert(reader.read_active() == false);
    std::remove(path.c_str());
}

static void test_contenido_basura_es_false() {
    const std::string path = "/tmp/hyperion_test_gpu_active_basura.txt";
    write_file(path, "xyz");
    GpuActiveFileReader reader(path);
    assert(reader.read_active() == false);
    std::remove(path.c_str());
}

// Simula la escritura atomica del lado Python (os.replace): el lector
// vuelve a leer el archivo en cada llamada, sin cachear -- un reemplazo
// entre dos llamadas debe verse reflejado de inmediato.
static void test_relee_en_cada_llamada() {
    const std::string path = "/tmp/hyperion_test_gpu_active_transicion.txt";
    write_file(path, "0");
    GpuActiveFileReader reader(path);
    assert(reader.read_active() == false);
    write_file(path, "1");
    assert(reader.read_active() == true);
    std::remove(path.c_str());
}

int main() {
    test_archivo_ausente_es_false();
    test_contenido_uno_es_true();
    test_contenido_cero_es_false();
    test_archivo_vacio_es_false();
    test_contenido_basura_es_false();
    test_relee_en_cada_llamada();
    std::printf("test_gpu_active_reader: OK\n");
    return 0;
}

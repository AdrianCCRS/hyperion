#include <cstdio>
#include <fstream>
#include <string>

#include "check.hpp"
#include "gpu_active_writer.hpp"

using namespace hyperion::gpu_loop;

static std::string slurp(const std::string& p) {
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int main() {
    const std::string dir = "/tmp/hyp_gpu_active_writer_test";
    std::string cmd = "rm -rf " + dir;
    CHECK(std::system(cmd.c_str()) == 0);
    const std::string path = dir + "/sub/gpu_active.txt";  // crea el directorio padre
    GpuActiveWriter w(path);
    CHECK(w.write(true));
    CHECK(slurp(path) == "1");
    CHECK(w.write(false));
    CHECK(slurp(path) == "0");
    CHECK(slurp(path + ".tmp").empty());  // el temporal no queda: se renombro
    // El lector C++ existente (cpu_loop/gpu_active_reader.hpp) parsea este mismo contenido: "1" activa, cualquier otra cosa no.
    CHECK(w.write(true) && slurp(path) == "1");
    cmd = "rm -rf " + dir;
    CHECK(std::system(cmd.c_str()) == 0);
    std::printf("gpu_active_writer_test: OK\n");
}

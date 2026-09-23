#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "decision_log.hpp"

using namespace hyperion::cpu_loop;

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// to_json() produce un objeto JSON plano con todos los campos del esquema,
// incluidos los opcionales ausentes como `null` -- el mismo esquema que
// decision_log.py debe poder deserializar sin campos faltantes.
static void test_to_json_campos_basicos() {
    DecisionRecord r;
    r.ts_ns = 123456789;
    r.arm = "sombra";
    r.device = "cpu";
    r.label = "memory_bound";
    r.confidence = 0.93f;
    r.features = {{"ipc", 1.5}, {"mpki", 2.0}};
    r.policy_action = "actuar";
    r.target_freq_khz = 3200000;
    r.applied_freq_khz = 0;
    r.written = false;
    r.write_failed = false;

    const std::string json = to_json(r);
    assert(contains(json, "\"ts_ns\":123456789"));
    assert(contains(json, "\"arm\":\"sombra\""));
    assert(contains(json, "\"device\":\"cpu\""));
    assert(contains(json, "\"label\":\"memory_bound\""));
    assert(contains(json, "\"confidence\":0.93"));
    assert(contains(json, "\"features\":{\"ipc\":1.5,\"mpki\":2}"));
    assert(contains(json, "\"policy_action\":\"actuar\""));
    assert(contains(json, "\"target_freq_khz\":3200000"));
    assert(contains(json, "\"written\":false"));
    assert(contains(json, "\"error\":null"));
    assert(contains(json, "\"inference_time_ns\":null"));
}

// Campos opcionales ausentes (tick abstenido: sin decision, sin error) se
// serializan como `null`, nunca como 0/"" fingiendo un valor real.
static void test_to_json_campos_ausentes_son_null() {
    DecisionRecord r;
    r.arm = "activo";
    r.device = "cpu";
    // label, confidence, error, inference_time_ns, actuation_time_ns: sin asignar (nullopt)
    const std::string json = to_json(r);
    assert(contains(json, "\"label\":null"));
    assert(contains(json, "\"confidence\":null"));
    assert(contains(json, "\"error\":null"));
    assert(contains(json, "\"actuation_time_ns\":null"));
}

// El escape de comillas/backslash no debe romper el JSON si algun dia un
// campo de texto (p.ej. error) trae caracteres especiales.
static void test_to_json_escapa_comillas() {
    DecisionRecord r;
    r.arm = "sombra";
    r.device = "cpu";
    r.error = std::string("mensaje con \"comillas\" y \\barra");
    const std::string json = to_json(r);
    assert(contains(json, "mensaje con \\\"comillas\\\" y \\\\barra"));
}

// DecisionLogWriter escribe una linea JSON por llamada a write(), en modo
// append -- verificado escribiendo dos registros y releyendo el archivo.
static void test_writer_append_una_linea_por_registro() {
    const std::string path = "/tmp/hyperion_test_decision_log.jsonl";
    std::remove(path.c_str());
    {
        DecisionLogWriter w(path);
        assert(w.is_open());
        DecisionRecord r1;
        r1.arm = "sombra";
        r1.device = "cpu";
        r1.ts_ns = 1;
        w.write(r1);
        DecisionRecord r2;
        r2.arm = "activo";
        r2.device = "cpu";
        r2.ts_ns = 2;
        w.write(r2);
    }
    std::ifstream in(path);
    std::string line;
    int n_lines = 0;
    while (std::getline(in, line)) {
        assert(!line.empty());
        n_lines++;
    }
    assert(n_lines == 2);
    std::remove(path.c_str());
}

int main() {
    test_to_json_campos_basicos();
    test_to_json_campos_ausentes_son_null();
    test_to_json_escapa_comillas();
    test_writer_append_una_linea_por_registro();
    std::printf("test_decision_log: OK\n");
    return 0;
}

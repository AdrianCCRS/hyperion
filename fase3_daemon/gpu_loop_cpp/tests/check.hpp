#pragma once
#include <cstdio>
#include <cstdlib>
// CHECK propio (no assert): el build por defecto es RelWithDebInfo, que define NDEBUG y volvería un assert() un no-op silencioso.
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK fallo %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

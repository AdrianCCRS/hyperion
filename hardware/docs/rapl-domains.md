## RAPL Domains (Intel Running Average Power Limit)

RAPL expone contadores de energía para distintos subsistemas del procesador a través de
`/sys/class/powercap/intel-rapl:*`. Cada dominio representa una región energética del chip.

### Dominios principales

| Dominio | Descripción | Uso en el proyecto |
|-------|-------------|----------------|
| `intel-rapl:0` | **Package domain**. Energía total consumida por el socket de CPU (cores, cache, memory controller, uncore). | Métrica principal para calcular **Energy–Delay Product (EDP)**. |
| `intel-rapl:0:0` | **PP0 (Core domain)**. Energía consumida solo por los núcleos de CPU. | Útil para analizar impacto directo de **DVFS en los cores**. |
| `intel-rapl:0:1` | **DRAM domain**. Energía consumida por el subsistema de memoria. | Útil para identificar fases **memory-bound**. |
| `intel-rapl:1` | Segundo **package domain** (en sistemas con múltiples paquetes o dominios adicionales). | Generalmente irrelevante en sistemas de un solo socket. |

### Dominios MMIO

| Dominio | Descripción |
|-------|-------------|
| `intel-rapl-mmio` | Implementación alternativa de RAPL usando **Memory-Mapped I/O** en lugar de registros MSR. |
| `intel-rapl-mmio:0` | Subdominio MMIO equivalente a un package domain. |

Estos dominios miden la misma energía que RAPL clásico, pero mediante un mecanismo de acceso distinto.

### Dominios relevantes para la tesis

Para la evaluación energética del agente DVFS se recomienda recolectar:

- **Package energy** (`intel-rapl:0`) → energía total CPU
- **DRAM energy** (`intel-rapl:0:1`) → análisis de comportamiento memory-bound

Los contadores de energía se leen desde:
`/sys/class/powercap/intel-rapl:*/energy_uj`
Unidad:
microjoules (µJ)
La energía consumida en un intervalo se calcula como:
E = (energy_uj_t2 - energy_uj_t1) / 1e6

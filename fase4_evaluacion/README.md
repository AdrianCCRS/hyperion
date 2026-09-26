# Fase 4 — Validación experimental

Cumple el **Objetivo 4**: evaluar el impacto empírico del agente vía EDP,
determinando si el ahorro compensa el overhead de inferencia frente a
gobernadores nativos de Linux. Ver
`Plan_Detallado_Realineacion_Hyperion.md` §5.

## Alcance de este módulo

La evaluación se ejecutó en paccaA100 con campañas dedicadas (`scripts/pacca/hyp_fase4_*.sbatch`
y `hyp_cloverleaf_*.sbatch`, un solo job por serie, sin `--time`) que lanzan los daemons de la
Fase 3 sobre aplicaciones compuestas con fases de verdad conocida (`fase3_daemon/composite_apps/`).
Este directorio contiene el análisis y las utilidades de esa evaluación:

| Archivo | Qué hace |
|---|---|
| `analyze_matrix.py` | Resume una campaña (`results.csv`, `phases.jsonl`, `*_decisions.jsonl`): medianas de tiempo y energía, EDP del nodo, razones frente a la base, pruebas estadísticas y puntuación de la clasificación contra las fronteras reales de fase |
| `gate_preflight_E.py` | Puerta previa del escenario E: comprueba las condiciones de la medición antes de gastar el nodo |
| `governors.py` | Conmutación y restauración verificada del gobernador de CPU |
| `edp_report.py`, `run_evaluation.py` | Reporte de EDP con significancia estadística a partir de `windows.csv` de campañas de kernels |

### Campañas y dónde quedan sus resultados

| Escenario | Job de Slurm / script | Datos versionados (`docs/libro/datos/fase4_20260926/`) |
|---|---|---|
| Matriz inicial (CPU, GPU y conjunto; vistos e inéditos) y su repetición independiente (C) | `hyp_fase4_performance.sbatch` (serie completa bajo el estado nativo del nodo, job 7696) y `hyp_fase4_matrix.sbatch` | `fase4_A.csv` (matriz), `fase4_C.csv` (repetición), `tabla_resultados.tex`, `tabla_C.tex`, `clasificacion.json` |
| E: GPU dominada por memoria (E-A vistas, E-B inéditas) | `hyp_fase4_E_run.sbatch`, `hyp_fase4_EA_confirmatorio.sbatch`, `hyp_fase4_EB_confirmatorio.sbatch`, `hyp_fase4_EB_faltantes.sbatch`, `hyp_fase4_myocyte_diag.sbatch` | `fase4_E.csv`, `fase4_EA.csv`, `fase4_EB.csv`, `fase4_EBdiag.csv` (diagnóstico, job 7706), `resumen_EA.json`, `E_inedito_detalle.json`, `tabla_E.tex` |
| D: LAMMPS | `hyp_fase4_D_calib.sbatch`, `hyp_fase4_D_probe.sbatch` | `fase4_D.csv`, `tabla_D.tex` |
| F: F1 fijo sobre fases de cómputo | `hyp_fase4_F_f1compute.sbatch` | (medición previa al confirmatorio E-A) |
| Base sin turbo (control de repetibilidad) | incluido en las series anteriores | `fase4_noturbo.csv`, `tabla_noturbo.tex` |
| CloverLeaf CUDA Fortran | `hyp_cloverleaf_confirm.sbatch`, `hyp_cloverleaf_shadow.sbatch` | `../fase4_20260924/cloverleaf_confirm_7692.csv`, `tabla_cloverleaf.tex` |

El libro reporta las razones de EDP en figuras; las tablas con valores absolutos (segundos y
julios por brazo) y las de clasificación por escenario se conservan como archivos `tabla_*.tex`
en esa carpeta. Ver `docs/libro/datos/README.md`.

La base de la Fase 4 final es el estado nativo del nodo con gobernador `performance`, turbo
desactivado y 3.2 GHz fijos en la CPU, con gestión automática de la GPU. Las campañas anteriores
(`docs/libro/datos/fase4_20260924/`) partían con turbo activo y quedan como registro histórico.

## Los 3+1 escenarios (§5.1)

1. `ondemand` / `schedutil` — gobernadores nativos reactivos de Linux.
2. `performance` — frecuencia fija de alto rendimiento.
3. El agente propuesto (`fase3_daemon/`).

⚠️ **Hueco de código que motivó `governors.py`, confirmado por la
auditoría exclusiva de código de esta reconstrucción**: antes de esta
reconstrucción, no existía ningún código que conmutara `scaling_governor`
a `ondemand`/`schedutil` explícitamente — solo un modo `native_governor`
en `freqctl.py` que significa "dejar lo que el nodo ya tuviera puesto". Sin
esto, el escenario 1 de la lista no tenía ningún soporte real.

## Procedimiento paso a paso para producir los datos de un escenario de gobernador

```python
from common.hpc import environment
from fase4_evaluacion.governors import governor_scenario
from fase1_telemetria import runner  # o invocar run_campaign.py como subproceso

env = environment.detect_environment()
with governor_scenario(env.delegated_cpus, "ondemand", env):
    # correr aquí la campaña completa del catálogo (run_campaign.py o
    # runner.run_single() por kernel), con output_dir propio por escenario
    ...
# al salir del bloque, el gobernador original queda restaurado -- verificado
# por relectura, incluso si el bloque lanzó una excepción
```

`governor_scenario()` valida contra `scaling_available_governors` del nodo
antes de tocar nada (`GovernorNotAvailableError` si el gobernador pedido
no está disponible) y restaura el original garantizado, con la misma
disciplina de escritura+verificación por relectura que el resto del
proyecto (`common.hpc.freqctl.set_governor`/`read_governors`, funciones
nuevas de esta reconstrucción, aditivas — no tocan ningún camino de código
existente de `freqctl.py`).

## Generar el reporte final

```bash
python3 fase4_evaluacion/run_evaluation.py \
    --scenario agente     ~/hyperion-results/campaigns/agente/*/windows.csv \
    --scenario performance ~/hyperion-results/campaigns/performance/*/windows.csv \
    --scenario ondemand    ~/hyperion-results/campaigns/ondemand/*/windows.csv \
    --scenario schedutil   ~/hyperion-results/campaigns/schedutil/*/windows.csv \
    --agent-scenario agente \
    --output fase4_evaluacion/reporte_final.txt
```

Un escenario sin `windows.csv` disponibles se omite del reporte con un
aviso explícito en stderr — nunca se fabrica una fila con datos que no
existen. La comparación se hace **por separado para cada (dispositivo,
clase)**, nunca como un único número agregado (exigencia explícita del
plan, §5.2) — un kernel se incluye en una comparación solo si tiene datos
tanto en el escenario del agente como en ese baseline específico.

## Significancia estadística

Cada fila del reporte usa `common.stats.paired_significance_test`
(Wilcoxon/t-test/Mann-Whitney, elegido automáticamente según normalidad de
las diferencias) sobre la mediana de EDP por kernel — la misma prueba que
usa `fase3_daemon/policy/derive_policy_table.py` para decidir la política,
para que "mejora estadísticamente defendible" signifique lo mismo en la
política que se despliega y en la evaluación que la juzga.

⚠️ **Hueco de código que motivó esto**: cero uso de `scipy.stats` en todo
el repositorio antes de esta reconstrucción (confirmado en la auditoría
exclusiva de código) — el análisis de EDP existente comparaba magnitudes
(razones), nunca producía un p-valor.

## Tests

```bash
python3 -m pytest fase4_evaluacion/tests/ -q
```

14 tests: 6 de `governors.py` (incluye conmutación y restauración con
sysfs simulado en disco real), 5 de `edp_report.py`, 3 de
`run_evaluation.py` (subprocess real contra `windows.csv` sintéticos en
disco, incluyendo el caso de escenario faltante).

## Limitaciones conocidas

- No hay orquestación automática de "correr el catálogo completo bajo todos los escenarios" en una
  sola invocación: cada escenario es una campaña propia de `scripts/pacca/`.
- `run_evaluation.py` compara campañas de kernels con `windows.csv`; las aplicaciones compuestas
  de la Fase 4 se analizan con `analyze_matrix.py`.
- La aplicación E-B con `myocyte` tiene una variabilidad de duración propia (~41 s o ~50 s) que
  impide comparar medianas de pocas repeticiones; ver `scripts/pacca/hyp_fase4_myocyte_diag.sbatch`.

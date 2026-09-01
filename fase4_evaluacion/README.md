# Fase 4 — Validación experimental

Cumple el **Objetivo 4**: evaluar el impacto empírico del agente vía EDP,
determinando si el ahorro compensa el overhead de inferencia frente a
gobernadores nativos de Linux. Ver
`Plan_Detallado_Realineacion_Hyperion.md` §5.

## ⚠️ Alcance real de este módulo

`run_evaluation.py` **genera el reporte de comparación a partir de
`windows.csv` ya producidos** — no orquesta automáticamente correr el
catálogo completo bajo los 3 escenarios en una sola invocación. Eso
requeriría dos piezas que no se construyeron en esta reconstrucción (documentado
en sus propios README, con la razón): que `fase1_telemetria/campaign.py`
acepte un wrapper de escenario de gobernador alrededor de cada corrida
(hoy no tiene ese punto de extensión), y que `fase3_daemon/` esté completo
(el loop de CPU real todavía no lo está).

Lo que sí está completo y probado: la conmutación real de gobernador
(`governors.py`) y el cálculo/reporte de EDP con significancia estadística
(`edp_report.py`).

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

- No hay orquestación automática de "correr el catálogo completo bajo los
  4 escenarios" en una sola invocación (ver arriba).
- El "overhead del agente" (tiempo de inferencia + actuación, §5.2) no se
  calcula todavía en este reporte — depende de que `fase3_daemon/` registre
  ese log estructurado en producción (§4.3 punto 10), lo cual a su vez
  depende del loop de CPU real, no construido todavía.

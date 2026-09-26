# Datos que respaldan el libro

Resultados ya procesados (CSV, JSON y tablas `.tex`) de los que salen las figuras y las tablas del
libro. Las figuras se regeneran con los scripts de `../scripts/`; los datos crudos de las campañas
(muestras y trazas por corrida) viven en el clúster y no se versionan.

El libro conserva en sus anexos solo las tablas indispensables. Todo el detalle numérico adicional
(rejillas por nivel y clase, intervalos de confianza, tablas por escenario de la Fase 4) está aquí.

## Carpetas

| Carpeta | Contenido | Estado |
|---|---|---|
| `cpu_calidad_30fam/` | Clasificador de CPU final (30 familias, 6 variables): inventario y muestreo (`book/`), contrato de variables (`contract/`), exactitud por familia y por nivel de frecuencia, calibración, umbral de abstención (`threshold_final_grid.csv`), curva de aprendizaje, latencia y política de frecuencia de CPU (`politica/`, `policy_by_level.csv`) | Vigente |
| `gpu_calidad_20260922/` | Conjunto histórico de GPU (483 corridas) y política de GPU por familia (`politica/policy_by_family.json`) | Vigente |
| `gpu_calidad_20260924/rf_operativo/` | Random forest de GPU con tres entradas invariantes: probabilidades LOFO, calibración, umbral por familia y celda, importancia por permutación | Vigente |
| `fase4_20260926/` | Fase 4 final (base sin turbo, job 7696) y las repeticiones posteriores de E-B: un CSV por escenario (`fase4_A`, `_C`, `_D`, `_E`, `_EA`, `_EB`, `_EBdiag`, `_noturbo`), `clasificacion.json` y las tablas `tabla_*.tex` con valores absolutos y razones frente a la base | Vigente |
| `fase4_20260924/` | Campañas anteriores de la Fase 4 (con turbo activo) y el confirmatorio de CloverLeaf (`cloverleaf_confirm_7692.csv`, `tabla_cloverleaf.tex`) | Histórico, salvo CloverLeaf |
| `cpu_calidad_20260918/`, `cpu_modelo_20260918/` | Iteración anterior del clasificador de CPU | Histórico, reemplazado por `cpu_calidad_30fam/` |

## Tablas de la Fase 4 que no están en el libro

Las tablas de duración y energía por brazo en `fase4_20260926/` se generan con
`../scripts/generar_figuras_fase4_20260926.py`. El libro reporta las razones en figuras y
conserva la tabla de restauración del agente; el resto queda aquí:

| Archivo | Contenido |
|---|---|
| `tabla_resultados.tex` | Matriz inicial: mediana de duración, energía de CPU y GPU, EDP y valor p por brazo |
| `tabla_C.tex` | Repetición independiente de CPU y conjunto (escenario C) |
| `tabla_E.tex`, `tabla_clasificacion_CE.tex` | Escenario E (E-A y E-B) y clasificación de fase contra las fronteras reales |
| `tabla_clasificacion.tex` | Clasificación de fase del agente en la matriz inicial |
| `tabla_D.tex` | LAMMPS: EDP por entrada y brazo |
| `tabla_noturbo.tex`, `tabla_activo_vs_sombra.tex` | Control de repetibilidad de la base y razón activo frente a sombra |

## Convenciones

- `REF` en los archivos es la base del libro (en la Fase 4 final: gobernador `performance`, turbo
  desactivado, 3.2 GHz fijos en la CPU). `base_noturbo` es una segunda medición de la misma base.
- Los JSON y logs de esta carpeta sí se versionan (el `.gitignore` general los excluye en el resto
  del repositorio).

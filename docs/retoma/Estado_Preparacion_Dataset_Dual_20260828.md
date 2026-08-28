# Estado de preparación del primer dataset CPU/GPU — 2026-08-28

Documento de relevo operativo. La fuente de verdad sigue siendo, en este
orden, el plan de trabajo para objetivos/alcance, el código y los manifiestos
vigentes para lo que se ejecuta, y la entrada ARC más reciente para decisiones
técnicas. No sustituye esas fuentes.

## 1. Estado inmediato

- Rama local y checkout de pacca: `fase-02`, commit `9d8e1b7`
  (`prepare cold-warm dual dataset campaigns`). El commit está publicado en
  `origin/fase-02`.
- Job activo al redactar este relevo: **6716**, smoke GPU integrado corregido,
  lanzado con
  `run_campaign_pacca_dual_coldwarm_gpu_smoke.sbatch`.
- No seguir sondeando el job automáticamente: el autor avisará cuando termine
  para ahorrar su cuota de uso.
- Si 6716 termina con 108/108 corridas aceptadas y restaura el estado, el paso
  siguiente autorizado es lanzar **primero la campaña CPU completa**. Auditarla
  antes de iniciar la primera sesión GPU completa.
- No hay un rediseño metodológico abierto que bloquee este primer intento. El
  constructor final del dataset de nivel 2 sí sigue pendiente y no debe
  confundirse con la adquisición de los crudos.

## 2. Decisiones metodológicas confirmadas por el autor

- Se recolecta para la estrategia **A** (decisión estática previa) y la
  estrategia **C** (la primera ejecución real informa las siguientes). Se
  descartó la estrategia B de sonda CPU sintética.
- Todos los operandos GPU empiezan en host y todos los resultados regresan al
  host. SpMV transfiere también la matriz CSR completa.
- Región **cold**, primaria: empieza antes de la primera llamada CUDA e incluye
  contexto, handles/planes/descriptores, asignaciones/workspace, H2D, primera
  operación, sincronización y D2H. El análogo CPU incluye inicialización
  perezosa real y primer cómputo.
- Región **warm**, suplementaria: reutiliza recursos, pero vuelve a transferir
  los datos de cada operación GPU. Cada repetición de campaña abre un proceso
  nuevo.
- Generación de entradas y verificación del resultado quedan fuera de ambas
  regiones.
- Se mantiene `n=3` como base para el **primer intento**, con suplementos
  dirigidos posteriores donde CV o margen EDP lo exijan; FFT ya es candidato.
  No aumentar silenciosamente todas las repeticiones.
- El subtotal energético comparable debe incluir RAPL package + DRAM + NVML en
  ambos ejes. En las corridas CPU, NVML mide la contribución ociosa de la GPU;
  no se reemplaza por cero.
- No cambiar director, plataforma experimental, marco legal ni estrategia
  multinodo sin confirmación explícita del autor.

## 3. Cambios implementados y publicados

Cadena principal de commits, del contrato temporal a la preparación final:

- `b549767`: contrato cold/warm en los kernels duales y runner.
- `96d6f5f`: checksums del catálogo.
- `e017e20`: parser robusto de marcadores y telemetría NVML en CPU.
- `537452a`: aislamiento GPU para el alcance energético CPU.
- `4e154f3`: primera llamada CUDA dentro de cold; no usar el preload shim en
  kernels duales.
- `a9a3c57`: iteraciones CPU ajustadas con datos del job 6696.
- `9eb3558`: I10 solo cuenta telemetría dentro de la región warm medida.
- `fda10d7`: snapshot/restauración del estado CPU completo.
- `d7c538e`: evita escribir redundantemente un governor no escribible.
- `0ba20df`: checksums de wrappers GPU reconstruidos.
- `0b799c9`: criterio de actividad GPU por potencia y sonda de reposo para la
  rejilla dual.
- `9d8e1b7`: manifiestos/launchers completos, timeout de 180 s, subtotal
  energético simétrico, mapa de reposo exacto y metodología reconciliada.

Archivos centrales ya listos:

- `orchestrator/schemas/campaigns/campaign_pacca_dual_cpu_full.yaml`
- `orchestrator/schemas/campaigns/campaign_pacca_dual_gpu_full.yaml`
- `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_cpu_full.sbatch`
- `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch`
- `scripts/pacca/gen_dual_campaign_manifests.py`
- `docs/general/metodologia_selector_cpu_gpu_20260827.md`

Los manifiestos generados cargan correctamente. La matriz real, construida con
catálogo, es:

- CPU: 1632 combinaciones, 2176 lanzamientos de proceso.
- GPU: 6528 combinaciones, 8704 lanzamientos de proceso.
- Total: 8160 combinaciones, 10880 lanzamientos.

Pruebas locales posteriores al último cambio:

- `python -m pytest -q tests/orchestrator`: **568 passed**.
- `python -m pytest -q tests/classifier`: **60 passed**.
- `bash -n` sobre ambos launchers completos: correcto.
- `git diff --check`: correcto.

La invocación monolítica de todo pytest presenta una colisión de descubrimiento
entre `tests/classifier` y el paquete `classifier`; las dos suites pasan por
separado. No mezclar ese arreglo de infraestructura con la campaña actual.

## 4. Evidencia experimental ya cerrada

### CPU

- Job 6696 antiguo: 1632 corridas; 1614 aceptadas y 18 I10 bajo las reglas
  antiguas. Sirvió para ajustar iteraciones, no es el dataset cold/warm final.
- Job 6710, smoke CPU corregido: **108/108 aceptadas**, marcadores válidos,
  frecuencia, RAPL y NVML presentes. Slurm lo marcó `FAILED 70` únicamente por
  una escritura redundante del governor durante la restauración; los datos son
  válidos y el defecto quedó corregido en `d7c538e`.
- Jobs 6711 y 6712: caos real con `SIGTERM`, ambos `COMPLETED 0`. Confirmaron
  salida 143 del hijo y restauración exacta de Turbo y de los 12 CPU lógicos.

### GPU

- Job 6713, primer smoke cold/warm GPU: 98 aceptadas y 10 I10. Los diez rechazos
  fueron GEMM/Cholesky N=64 válidos que el umbral legado
  `gpu_util_pct >= 5 %` no veía. No hubo C03, timeout, checksum ni divergencia
  de frecuencia. El resultado se conservó en:
  `/home/latorresn/hyperion-results/campaigns/pacca_dual_coldwarm_gpu_smoke_20260828__legacy_util_floor_6713`.
- Job 6714, reposo de la rejilla exacta: `COMPLETED 0`, 9:27, stderr vacío,
  300 muestras durante 60 s por nivel. El reloj observado coincidió exactamente
  con el solicitado:

| Nivel | MHz observado | media reposo (mW) | p95 (mW) | máximo (mW) |
|---|---:|---:|---:|---:|
| REF | 210 en reposo/nativo | 34837.9 | 35050 | 35200 |
| F0 | 1410 | 56565.5 | 56950 | 57980 |
| F1 | 1215 | 45052.4 | 45360 | 46100 |
| F2 | 1005 | 38466.0 | 38720 | 39170 |
| F3 | 810 | 36941.6 | 37120 | 37680 |
| F4 | 615 | 35369.0 | 35550 | 35990 |
| F5 | 405 | 34837.9 | 34950 | 35190 |
| F6 | 210 | 34368.9 | 34500 | 34600 |

Artefacto:
`/home/latorresn/hyperion-results/analysis/gpu_idle_power_dual_6714.out`.

El manifiesto usa esas medias como línea de reposo. Los márgenes de actividad
REF/F0/F3/F6 conservan anclajes de ARC-194; F1/F2/F4/F5 están marcados como
interpolación lineal por MHz. No se presentan como nuevas mediciones bajo
carga. Con las nuevas líneas, una revalidación offline de las 108 corridas de
6713 produjo **108/108 aceptables, cero rechazos**.

## 5. Auditoría obligatoria al terminar 6716

Conexión: `ssh hpc-unicartagena`, luego `ssh pacca`. La cuenta es compartida:
no tocar archivos ajenos; cualquier diagnóstico nuevo va en `~/yacacerest/`.

1. Confirmar Slurm, final normal y ausencia de error:

   ```bash
   sacct -j 6716 --format=JobID,State,ExitCode,Elapsed -n -X
   tail -n 80 /home/latorresn/hyperion-results/campaigns/dual_cw_gpu_6716.out
   tail -n 80 /home/latorresn/hyperion-results/campaigns/dual_cw_gpu_6716.err
   ```

   Se espera `COMPLETED 0:0`, `CAMPAIGN_SCRIPT_DONE`, stderr sin error y líneas
   finales de restauración CPU/Turbo. Un warning CAL07 conocido no invalida por
   sí solo la campaña.

2. Leer
   `/home/latorresn/hyperion-results/campaigns/pacca_dual_coldwarm_gpu_smoke_20260828/campaign_metadata.json`.
   Debe haber **108 aceptadas y 0 rechazadas**, y
   `frequency_restored_verified: true` al cierre.

3. Auditar, no solo contar veredictos:

   - 108 `dispatch_timing_contract_valid == true`.
   - Orden temporal
     `cold_t0 <= setup_complete <= cold_t1 <= warm_t0 <= warm_t1`.
   - Ningún C03, timeout, checksum o divergencia de reloj.
   - Cada corrida con al menos 5 ventanas GPU etiquetadas dentro de warm.
   - RAPL package/DRAM y muestras NVML presentes.
   - Frecuencias GPU observadas correctas para REF/F0/F6.
   - Estado final CPU idéntico al inicial: governor `performance`, mínimos
     800000, máximos 3600000 en `0-5,16-21`, y `no_turbo=0`.
   - GPU restablecida a reloj nativo. El launcher también ejecuta `-rgc` antes
     de cada campaña, pero eso no sustituye comprobar la restauración.

4. Revisar distribución de potencia sobre el piso, especialmente GEMM y
   Cholesky N=64. El objetivo es confirmar que dejaron de ser falsos I10 sin
   relajar otros gates.

Si cualquiera de estos puntos falla, **no lanzar la campaña completa** hasta
explicar la causa. No borrar la corrida fallida; preservarla con un sufijo que
incluya el job y el motivo.

## 6. Lanzamiento siguiente si 6716 pasa

Primero verificar que el directorio nuevo no exista o esté realmente vacío:

```bash
test ! -e /home/latorresn/hyperion-results/campaigns/pacca_dual_cpu_full_20260828
cd /home/latorresn/hyperion
sbatch orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_cpu_full.sbatch
```

La campaña CPU completa solicita GPU y nodo exclusivo porque NVML forma parte
del subtotal energético. Esperado: 1632 combinaciones, aproximadamente 2.5 h,
límite Slurm de 4 h y timeout interno de 13200 s para dejar 20 minutos de
margen de restauración. No encadenar automáticamente la campaña GPU: auditar
primero aceptación, tiempos, cobertura, energía, actuación y restauración CPU.

Después, la GPU completa se ejecuta en sesiones reanudables:

```bash
cd /home/latorresn/hyperion
sbatch orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch
```

Cada sesión tiene 5:30 h de Slurm y timeout interno de 18000 s, dejando 30 min
para salida limpia y restauración. Relanzar el mismo script reanuda el mismo
`output_dir`; la primera sesión crea calibraciones y las siguientes las cargan.
La proyección completa es 48–55 h repartidas en varias sesiones, nunca un único
job de dos días. Auditar el progreso y los rechazos entre sesiones.

## 7. Riesgos y límites que no deben ocultarse

- `classifier/features/pair_dataset.py` es legado e incompatible con el
  contrato cold/warm y el subtotal energético simétrico. **No usarlo** para
  construir el dataset final. Los crudos conservan marcadores y contadores;
  falta implementar el agregador de nivel 2.
- En regiones CPU cold muy cortas puede no caer una muestra completa dentro del
  intervalo. El tiempo sigue siendo exacto por marcadores y existen muestras de
  energía que lo acotan, pero la energía requerirá integración por solapamiento
  y una bandera de baja resolución; no inventar precisión.
- Las líneas de reposo F0..F6 sí fueron medidas en el reloj exacto. Algunos
  márgenes de actividad intermedios son interpolados a partir de evidencia
  previa; mantener esa distinción epistémica.
- CAL-10/D04 puede advertir CV alto cuando la media del contador es casi cero.
  Es warning, no permiso para cambiar umbrales sin evidencia.
- La campaña GPU se valida corrida a corrida. Si aparecen rechazos sistemáticos
  en niveles intermedios, detener tras la auditoría de la sesión y diagnosticar;
  no bajar márgenes retrospectivamente para hacer pasar datos.

## 8. Higiene del repositorio y de pacca

- Cambios locales ajenos preservados deliberadamente:
  `docs/libro/main.pdf` modificado y `scripts/pacca/inspect_c03.py` no
  versionado. No incluirlos en commits ni eliminarlos sin consultar al autor.
- En pacca existen archivos RAJAPerf no versionados de la cuenta compartida;
  no tocarlos.
- Stashes remotos conservados:
  `catalog-checksums-rebuild-6709`,
  `catalog-checksums-recorded-in-96d6f5f` y
  `catalog-checksums-before-coldwarm-rebuild-20260828`.
- No borrar corridas rechazadas. Se archivan con nombre explícito y se conservan
  como evidencia.


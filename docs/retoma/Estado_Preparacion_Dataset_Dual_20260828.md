# Estado de preparación del primer dataset CPU/GPU — 2026-08-28

Documento de relevo operativo. La fuente de verdad sigue siendo, en este
orden, el plan de trabajo para objetivos/alcance, el código y los manifiestos
vigentes para lo que se ejecuta, y la entrada ARC más reciente para decisiones
técnicas. No sustituye esas fuentes.

## 1. Estado inmediato

- Rama `fase-02`. El código de las campañas ejecutadas corresponde al commit
  `9d8e1b7`; los commits posteriores hasta este relevo solo actualizan esta
  documentación operativa.
- El smoke GPU 6716 y la campaña CPU completa 6718 terminaron y pasaron sus
  auditorías. La primera sesión GPU completa, **job 6721**, fue cancelada por
  solicitud del autor a los 22:57 para liberar paccaA100 a otros usuarios.
  Conservó 119 corridas aceptadas y cero rechazadas. Reanudar mañana con el
  mismo launcher, después de confirmar disponibilidad del nodo.
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

- Job 6716, repetición integrada con el criterio corregido: `COMPLETED 0:0` en
  12:34, **108 aceptadas, 0 rechazadas**, matriz completa y
  `frequency_restored_verified: true`. La auditoría corrida por corrida encontró
  cero anomalías: 108 contratos temporales válidos y ordenados, mínimo 15
  ventanas warm útiles por corrida, RAPL/NVML y checksums presentes, y muestras
  CPU/GPU encerrando cold en 108/108. El peor warm fue 9.2955 s, lejos del
  timeout de 180 s. Los cinco mensajes de stderr fueron únicamente CAL-07 no
  bloqueantes ya documentados.
- Job 6717, comprobación física posterior de un segundo: `COMPLETED 0:0`, GPU
  ociosa a 210 MHz con máximo nativo 1410 MHz y stderr vacío. La relectura
  directa CPU posterior confirmó governor `performance`, 800000–3600000 en
  `0-5,16-21` y `no_turbo=0`.

## 5. Auditoría de 6716 — completada

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

Todos estos puntos pasaron. La campaña CPU completa se lanzó como job 6718.

## 6. Campaña CPU completa cerrada y primera sesión GPU activa

Antes del lanzamiento se confirmó que el directorio nuevo no existía. Comando
ya ejecutado:

```bash
cd /home/latorresn/hyperion
sbatch orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_cpu_full.sbatch
```

Slurm asignó el **job 6718**. Entró con CPU en `performance`, rango
800000–3600000 y `no_turbo=0`; el wrapper cambió correctamente al rango no-Turbo
800000–3200000 antes de la campaña y restauró el estado inicial exacto al
cierre.

La campaña superó preflight y comenzó la matriz con aceptaciones. Emitió
`CAL-10/D04: cv_pct=16.15 %`; se auditó antes de dejarla continuar: el CV de
IPC es solo **0.074 %**, mientras que el máximo proviene de `miss_rate`
(16.145 %) y MPKI (16.094 %) con medias casi nulas (`miss_rate≈0.00158`,
`MPKI≈0.0246`). Es la advertencia conocida por denominador cercano a cero, no
inestabilidad del cómputo de referencia ni razón para cambiar umbrales durante
la corrida.

Resultado final de 6718: `COMPLETED 0:0` en **02:35:36**, 1632 aceptadas,
0 rechazadas, matriz completa y `frequency_restored_verified: true`. Auditoría
de cobertura:

- 68 kernels × 8 niveles × 3 repeticiones, exactamente una corrida por triple;
- 204 corridas por nivel, 24 por kernel y 544 baselines de repetición 1;
- 1632 contratos temporales válidos y checksums presentes;
- mínimo 1062 ventanas warm etiquetadas por corrida, frente al requisito de 5;
- RAPL válido y muestras NVML presentes en 1632/1632;
- niveles aplicados exactamente: F0=3200, F1=2800, F2=2400, F3=2000,
  F4=1600, F5=1200 y F6=800 MHz;
- peor cold 1.0647 s y peor warm 6.9353 s, lejos del timeout de 180 s;
- relectura física posterior: CPU `performance` 800000–3600000,
  `no_turbo=0`; job 6720 confirmó GPU nativa, 210 MHz ociosa y máximo
  1410 MHz.

La auditoría detectó un efecto del observador en NVML: con sondeo cada 5 ms,
la GPU permaneció a 1410 MHz y la potencia cruda durante candidatos CPU quedó
en torno a 50–68 W aun con utilización 0 %. Antes y después de la campaña, y
en el job 6714 con sondeo espaciado, el reposo nativo real fue ~35 W. Por ello,
el agregador de nivel 2 no debe integrar la serie NVML cruda del eje CPU como
si fuera consumo de la aplicación. Para el subtotal CPU debe usar
`RAPL package+DRAM + 34.8379 W × duración de la región`; sigue contando la GPU
ociosa, no la reemplaza por cero. Conservar la serie cruda para auditar el
overhead instrumental. Esta corrección no afecta tiempos, PMC, uncore ni RAPL.

La primera sesión GPU completa se lanzó como **job 6721** y fue cancelada para
liberar el nodo compartido:

```bash
cd /home/latorresn/hyperion
sbatch orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch
```

Cada sesión tiene 5:30 h de Slurm y timeout interno de 18000 s, dejando 30 min
para salida limpia y restauración. Relanzar el mismo script reanuda el mismo
`output_dir`; la primera sesión crea calibraciones y las siguientes las cargan.
La proyección completa es 48–55 h repartidas en varias sesiones, nunca un único
job de dos días. Auditar el progreso y los rechazos entre sesiones.

Estado parcial de 6721 al cancelar:

- Slurm: `CANCELLED` a los 22:57 de ejecución.
- 119 aceptadas, 0 rechazadas, 1.376 core-hours y 525 MiB preservados.
- Las 119 cubren los cuatro niveles CPU, los ocho niveles GPU, las seis
  operaciones y las tres repeticiones.
- Mínimo 17 ventanas warm válidas por corrida; peor cold 2.241 s y peor warm
  16.684 s, todavía lejos del timeout de 180 s.
- RAPL, NVML y uncore presentes y encerrando cold en 119/119.
- Los ocho pisos/márgenes GPU funcionaron en vivo. Exceso mínimo observado
  sobre el reposo: F0 5.357 W, F1 3.217 W, F2 2.783 W, F3 2.421 W,
  F4 1.993 W, F5 1.453 W, F6 1.024 W y REF 26.876 W; ninguno quedó por debajo
  de su margen declarado.
- La corrida interrumpida
  `dual_gemm_gpu_N512__REF__gpuF5__rep01` dejó solo `stdout.txt` y
  `stderr.txt`, sin metadata ni veredicto. No cuenta como aceptada; al reanudar
  se vuelve a ejecutar y los dos archivos se abren con truncado.
- Como Slurm cortó antes del cierre normal, `frequency_restored_verified`
  quedó `None`. La CPU se releyó ya restaurada (`performance`, 800000–3600000,
  `no_turbo=0`). El job de limpieza 6723 ejecutó `nvidia-smi -rgc` y confirmó
  GPU nativa con máximo 1410 MHz y stderr vacío. El nodo quedó libre.

Para reanudar mañana no crear un manifiesto ni un directorio nuevo. Ejecutar
exactamente el mismo launcher; CAM-11 carga las calibraciones ya completas,
salta las 119 aceptadas y continúa la matriz:

```bash
cd /home/latorresn/hyperion
sbatch orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_full.sbatch
```

## 7. Riesgos y límites que no deben ocultarse

- `classifier/features/pair_dataset.py` es legado e incompatible con el
  contrato cold/warm y el subtotal energético simétrico. **No usarlo** para
  construir el dataset final. Los crudos conservan marcadores y contadores;
  falta implementar el agregador de nivel 2.
- La serie NVML de las corridas CPU contiene el efecto de despertar la GPU por
  sondeo a 5 ms. No usarla directamente para el EDP CPU; aplicar la línea de
  reposo nativa medida como se documenta en §6. Esto permite salvar la campaña
  sin atribuir al candidato CPU la energía causada por el instrumento.
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

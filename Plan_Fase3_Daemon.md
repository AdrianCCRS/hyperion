# Plan de ejecución — Fase 3 (daemon de control DVFS)

Complementa `Plan_Detallado_Realineacion_Hyperion.md` §4. Donde este
documento y aquel discrepen, **manda este**: el plan de realineación se
escribió antes de tener los resultados de Fase 2, y dos de sus supuestos
quedaron refutados por medición (§0.2 y §0.3 de abajo).

Estado de partida (2026-09-22): `fase3_daemon/` tiene 31 pruebas Python y
1 de C++ en verde, con el loop de GPU completo, el actuador, la máquina de
decisión de CPU y `run_daemon.py` operando en `--dry-run`. El plan de
realineación afirma que esta pieza "no tiene ni un esqueleto parcial";
eso ya no es cierto y no debe usarse para dimensionar el trabajo restante.

---

## 0. Decisiones de diseño (Bloque 0)

### 0.1 Qué hace el daemon dado que la política de CPU es "no actuar"

**Contexto.** La tabla de política de CPU resultó *no actuar* en ambas
clases, y la de GPU *actuar en `memory_bound` a F1* (+8.9% de EDP) pero
con el candado de reloj averiado. Un daemon que solo traduzca "clase →
nivel" sería, hoy, un no-op en CPU.

**Decisión.** El daemon no se escribe para ejecutar la tabla de política y
ya; se escribe para **poder ejecutar el experimento de Fase 4**, que es lo
que da valor a la tabla (incluso a una tabla que dice "no actuar"). El
experimento tiene dos ejes.

**Eje 1 — la carga.** La tabla de política se derivó sobre corridas de un
solo kernel, donde ningún nivel *fijo* gana sobre REF. Una aplicación HPC
compuesta, que alterna fases de distinto régimen, es una pregunta
distinta y es justamente la premisa de la tesis: un agente que conmuta por
fase puede ganarle a cualquier nivel fijo aunque ningún nivel fijo le gane
a REF. Se construyen a mano aplicaciones compuestas de varios kernels:

- **A. Compuesta con familias conocidas** — kernels que sí estuvieron en
  el entrenamiento del clasificador.
- **B. Compuesta con familias inéditas** — kernels que nunca entraron en
  ninguna decisión de ajuste. Es el equivalente, a nivel de aplicación,
  del protocolo *leave-one-familia-out* con el que se validó el modelo: si
  A gana y B no, el resultado es memorización, no generalización.

**Eje 2 — el brazo de control.** Tres brazos, no dos, porque "sin actuar"
es ambiguo y esconde dos efectos distintos:

| Brazo | Daemon | Escribe frecuencia | Qué aísla |
|---|---|---|---|
| **base** | apagado | no | Línea base pura: gobernador nativo |
| **sombra** | encendido | no | **Costo** del agente: clasifica y decide en cada ciclo, sin actuar. `sombra − base` = sobrecarga de inferencia (objetivo 2) |
| **activo** | encendido | sí | Efecto total. `activo − sombra` = **beneficio** de la actuación, ya descontada su propia sobrecarga |

Sin el brazo *sombra*, cualquier ganancia de *activo* frente a *base*
mezcla el beneficio de actuar con el costo de inferir, y el objetivo 2
(que la sobrecarga no anule el beneficio) no se puede responder por
separado. Es una corrida más por aplicación, y es barata.

**Expectativa declarada de antemano, para no ajustarla después.** En CPU,
el piso de potencia medido (85 W, 84% del total a 3.2 GHz) acota el
oráculo por kernel a 2.2% de ganancia media. Una aplicación compuesta
puede superar a cualquier nivel fijo, pero no puede escapar de ese piso:
**no se espera una ganancia grande en CPU**, y un resultado plano es un
hallazgo válido, no un fallo del daemon. Lo que sí debe quedar medido es
la sobrecarga (brazo *sombra*) y el comportamiento de conmutación. En GPU
la expectativa es la contraria, porque ahí sí hay margen medido.

**Requisitos que esto impone al daemon** (todos verificables sin cluster):

1. **Brazo como parámetro de primera clase**, no un `--dry-run` de
   conveniencia: en el brazo *sombra* el daemon debe ejecutar exactamente
   el mismo trabajo que en *activo* (leer, construir el vector, inferir,
   consultar la tabla, decidir) y detenerse justo antes de la escritura.
   Si *sombra* saltara la inferencia, no mediría nada útil.
2. **Registro estructurado por decisión, idéntico en los tres brazos**:
   marca de tiempo, variables leídas, clase inferida, confianza, decisión
   de la tabla, si se escribió, tiempo de inferencia y tiempo de
   actuación. Es el insumo directo de la Fase 4 y del objetivo 4.
3. **Fronteras de fase conocidas en las aplicaciones compuestas.** Al
   construirlas nosotros, cada transición de kernel queda registrada con
   su marca de tiempo. Eso permite puntuar la **clasificación** contra una
   verdad conocida, no solo comparar EDP agregado — si el EDP no mejora,
   hay que poder distinguir "clasificó bien pero no había margen" de
   "clasificó mal".
4. **GPU lista aunque no se pueda probar.** Toda la ruta de GPU
   (detección de fase, inferencia, consulta de tabla, decisión,
   histéresis) se implementa y se valida en *sombra*, con la escritura del
   reloj como el único paso detrás de la bandera de brazo. Cuando H1 se
   repare, el brazo *activo* de GPU corre sin cambios de código.
5. **Restauración y caos** cubriendo los tres brazos (en *base* y
   *sombra* no hay nada que restaurar, y eso también debe verificarse:
   que el daemon no dejó rastro).

### 0.2 La medida defensiva de §4.1 queda refutada — CERRADA

§4.1 del plan de realineación prescribe: *"mientras `gpu_util_pct` reporte
actividad, forzar el reloj de CPU al mínimo"*, bajo el supuesto explícito
de que *"si la CPU está de verdad bloqueada esperando, bajar su reloj casi
no afecta el consumo"*.

Ese supuesto está **refutado por medición** (F1-XDEV-006, 2026-09-13):
fijar la CPU al mínimo durante una carga GPU la alarga **63-66%**
(134/134 y 1632/1632 pares más lentos en dos campañas), con la fase en
régimen estacionario casi duplicada (+93.8%) y una degradación que crece
con el tamaño del problema.

**Semántica corregida, ya implementada** en
`fase3_daemon/cpu_loop/include/cpu_phase_controller.hpp`: la señal de
coordinación pasa de ser un *objetivo* (bajar a él) a ser una *barrera*
(nunca pedir por debajo de él). Concretamente:

- Con la GPU activa, una petición de la política por debajo de
  `gpu_active_floor_khz` se **eleva** hasta ese valor.
- Si la política de la clase es "no actuar", no se escribe nada, ni
  siquiera con la GPU activa: la barrera solo puede elevar una petición
  que la política ya decidió hacer, nunca introduce una escritura propia.

El shim de blocking-sync (`common/hpc/native/blocking_sync_shim.cpp`)
sigue siendo la parte válida del mecanismo original: evita que una espera
bloqueante se lea como IPC alto ante el clasificador. Lo que se retira es
bajar el reloj por esa señal.

### 0.3 Un solo derivador de política, y que sea el correcto

Existen dos derivadores y **no coinciden**:

| | `fase3_daemon/policy/derive_policy_table.py` | `fase2_clasificador/analysis/{cpu,gpu}_policy_table.py` |
|---|---|---|
| Unidad de EDP | **ventana** (~1 ms) | **corrida completa** |
| IC95 por bootstrap | no | sí |
| Efecto mínimo relevante | no | sí |
| Des-duplicación por familia | no | sí (complemento por familia) |
| Resuelve la frecuencia física | sí (`resolved_*`) | **no** |
| Produjo los resultados del libro | no | sí |

El EDP por ventana es la unidad que el libro **rechaza explícitamente**:
con ventanas de duración fija, a frecuencia baja una ventana cubre menos
trabajo que a frecuencia alta, así que `E×Δt` a Δt fijo mide potencia, no
energía-retardo del trabajo. El derivador de Fase 3 produciría, en
silencio, una tabla distinta de la que sustenta el libro.

**Decisión.** Se retira `fase3_daemon/policy/derive_policy_table.py` como
derivador. La derivación vive en Fase 2 (como ya dice el libro en
§`sec:metodologia-politica-cpu`) y Fase 3 solo **carga y valida**.

**Dos defectos del artefacto canónico que hay que cerrar antes** (ambos
detectados al revisar el consumo real del daemon):

1. **`chosen_level` de GPU dice `F2`; el libro defiende `F1`.** El JSON
   sale del análisis por kernel, donde F2 aparece como el mejor nivel
   significativo; pero al des-duplicar los tres algoritmos medidos en dos
   tamaños (`dual_axpy`, `dual_spmv`, `dual_stencil`), F2 pierde
   significancia y F1 es el nivel que se sostiene. El artefacto debe
   reflejar la decisión por familia, que es la defendida.
2. **Falta `resolved_clock_mhz`/`resolved_freq_khz`.** Sin la frecuencia
   física resuelta, `build_controller_from_policy()` lanza `ValueError` y
   el daemon no puede actuar. La tabla debe ser autocontenida: el daemon
   no debe resolver un ID de nivel contra un manifiesto de campaña.

---

## 1. Bloques de trabajo

### Bloque A — Sin dependencia de hardware — CERRADO (2026-09-22)

- [x] **A1/A2.** `fase3_daemon/policy/build_policy_table.py` combina
  `policy_cpu.json` (kernel) con `policy_by_family.json` de GPU (familia,
  des-duplicado) y resuelve la frecuencia real (F1 → 1260 MHz). Verificado
  contra `build_controller_from_policy()` real. `derive_policy_table.py`
  retirado. Artefactos GPU comprometidos en
  `docs/libro/datos/gpu_calidad_20260922/politica/`.
- [x] **A3.** Candidato GPU vigente (regresión logística, 483 corridas, 16
  familias, 5 variables) exportado a `fase2_clasificador/models/`. El
  `.joblib` de 15-sep quedó retirado. Latencia: la cifra de `paccaA100` se
  preserva en la metadata con su procedencia documentada; no se remidió
  localmente (número sin sentido fuera del hardware de destino).
- [x] **A4.** `gpu_loop/classifier.py::HistoricalGpuClassifier` cablea el
  modelo real. Encontrado y resuelto en el camino: (a) el modelo se
  entrenó con mediana+std de múltiples muestras NVML por corrida, el loop
  solo daba una instantánea por evento — resuelto con un buffer móvil
  (`activity_poller` gana un `on_sample` aditivo), documentado como
  aproximación causal, no la definición exacta de entrenamiento; (b) el
  modelo predice booleano, no la string de la etiqueta.
- [x] **A5.** `fase3_daemon/cpu_loop/export_onnx.py`: ONNX verificado fila
  a fila contra las 1 169 833 filas reales de `tmp/cpu_quality_20260918/
  source.csv` (no una muestra): `max_diff_proba=2.98e-07`, 0 discrepancias.

49/49 tests de `fase3_daemon` en verde (36 previos del Bloque 0 + 13 nuevos).

#### Límite conocido de A4: dos capas de generalización sin validar

La tabla de política no está indexada por kernel (solo tiene 2 entradas,
`compute_bound`/`memory_bound`); que "funcione" con un kernel nunca visto
depende enteramente de que el **clasificador** generalice, no de que la
tabla lo conozca. Hay evidencia real de eso (LOFO 0.792, validación
externa sellada 0.779 sobre 4 familias nunca tocadas), pero esa evidencia
cubre solo una de dos capas necesarias, y de las otras dos no hay ninguna
medición todavía:

1. **Generalización del clasificador** (¿acierta la clase?) — medida:
   ~0.79-0.85 según cobertura. Evidencia fuerte.
2. **Generalización de la propia tabla** (¿la ganancia de F1 en
   `memory_bound` se sostiene en un kernel `memory_bound` nuevo?) — la
   validación LOFO de la Tabla \ref{tab:gpu-politica-ic} da 6.5% de
   ganancia realizada fuera de muestra con **n=8 familias** y un IC95 que
   casi toca cero ($-4.0$ a $13.1\%$). Dirección correcta, evidencia
   débil por tamaño de muestra pequeño, no por defecto de método.
3. **Aproximación del buffer móvil** (`HistoricalGpuClassifier`, A4b):
   sustituye la mediana/desviación estándar de la corrida completa con la
   que se entrenó el modelo por la de una ventana causal de ~20 muestras
   recientes. **Nunca se comparó contra la agregación real de
   entrenamiento** -- no hay ninguna cifra de cuánto se degrada el 0.792
   de LOFO al usar esta versión online en vez de la offline.
4. **Kernels que cruzan el punto de inflexión Roofline** (`rajaperf_gemm`,
   `rodinia_lud`, sección \ref{sec:resultados-gpu-gemm-evolucion} del
   libro): su clase real cambia con el nivel de frecuencia. Un kernel
   nuevo con ese comportamiento probablemente se clasifica mal, sin forma
   de saberlo de antemano con las cinco variables actuales.

No es un defecto de diseño del daemon: es exactamente la pregunta que la
aplicación compuesta **B (familias inéditas)** de §0.1 está pensada para
responder de forma empírica, en vez de asumir que el 0.792 de LOFO se
traduce directamente en beneficio real. Un resultado donde "el
clasificador acierta pero la ganancia no se sostiene" sería un hallazgo
válido de Fase 4, no un fallo del daemon -- pero hasta que esa medición
exista, el brazo *activo* de GPU sobre un kernel fuera del catálogo de
entrenamiento se apoya en una cadena de tres aproximaciones no verificadas
apiladas sobre una que sí lo está.

### Bloque B — Loop de CPU en C++

Integrar en `telemetry/`: leer el snapshot de `collector.hpp` por tick,
armar las 6 variables, inferir con ONNX, consultar la tabla, decidir con
`CpuPhaseController`. **Medir la latencia de inferencia antes de
integrarla al camino caliente**: si no cabe en el presupuesto de ~1 ms, la
decisión de diseño cambia (decidir cada N ticks), y eso es un resultado a
documentar, no un ajuste silencioso.

### Bloque C — Brazos de experimento e integración

- **C1.** Brazo como parámetro (`base`/`sombra`/`activo`) con registro
  idéntico en los tres (§0.1, requisitos 1 y 2).
- **C2.** Aplicaciones compuestas A y B, con fronteras de fase
  registradas (§0.1, requisito 3). Entran al catálogo con el mismo rigor
  que Fase 1: binario, suma de verificación, warmup calibrado.
- **C3.** Detección de fase de GPU probada de punta a punta contra un
  kernel real de terceros (hoy solo hay pruebas unitarias del sondeo).
- **C4.** Señal de coordinación probada entre ambos loops.
- **C5.** Prueba de caos de restauración, en los tres brazos.
- **C6.** Sobrecarga del daemon medida y registrada.
- **C7.** Modos `cpuset`/`pid`: hoy son banderas aceptadas sin wiring.

### Bloque D — Bloqueado por H1

Brazo *activo* de GPU. No se arranca hasta que el candado de reloj esté
reparado. Todo lo demás corre en *sombra* mientras tanto.

---

## 2. Criterios de cierre

Los del checklist §8 del plan de realineación, más los de §0.1:

- [ ] Ambos loops funcionando y verificados por separado.
- [ ] Detección de fase de GPU probada contra un kernel real de terceros.
- [ ] Señal de coordinación probada, con la semántica corregida de §0.2.
- [ ] Restauración verificada con prueba de caos real.
- [ ] Sobrecarga del daemon medida y registrada.
- [ ] Los tres brazos corren sobre las aplicaciones compuestas A y B, con
      registro por decisión suficiente para puntuar clasificación (no solo
      EDP) contra las fronteras de fase conocidas.
- [ ] Ruta de GPU completa y validada en *sombra*, de modo que el brazo
      *activo* no requiera cambios de código cuando H1 se repare.
- [ ] Aplicación B (familias inéditas) mide, sobre las decisiones
      reales del daemon, las tres capas de generalización sin validar
      documentadas en A4: exactitud del clasificador con el buffer móvil
      real (no la agregación offline de entrenamiento), y si la ganancia
      de F1 en `memory_bound` (n=8, IC95 casi en cero) se sostiene fuera
      del catálogo de política. Resultado negativo en cualquiera de las
      dos es un hallazgo válido, no un bloqueador de cierre.

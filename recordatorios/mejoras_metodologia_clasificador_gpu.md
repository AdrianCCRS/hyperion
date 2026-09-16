# Posibles mejoras a la metodología del clasificador GPU

Contexto: el F1 bajo de `compute_bound` (0.197 por-pliegue) tiene una causa
estructural, no solo el desbalance de clase: la intensidad operacional (OI)
de cada kernel se mide una sola vez, offline, con `ncu` (ver
`fase1_telemetria/postprocess.py:838-913`, comentario ARC-80), y queda
constante en todas las corridas de ese kernel. Como el ridge apenas se
desplaza entre niveles de frecuencia, 13 de 16 kernels utilizables resultan
100% de una sola clase, y en *leave-one-familia-out* eso deja 8 de 11
pliegues con soporte cero para una de las dos clases (F1 forzado a 0 por
convención de scikit-learn en esos pliegues, sin que eso refleje un error
real del modelo).

Menú de opciones para atacarlo, de más barata a más cara:

1. **Recalcular `scale_pos_weight` de XGBoost por pliegue** en vez de una
   sola vez global (0.2797 fijo hoy). Cambio de código puro
   (`fase2_clasificador/training/train_phase_gpu.py`, alrededor de la
   línea 244), no requiere clúster. No cambia el modelo ganador
   (`arbol_prof1` sigue mejor en F1 y latencia) pero hace la comparación
   entre los 7 modelos más justa.

2. **Reportar F1 macro adicional, promediado solo sobre los pliegues
   mixtos** (`rodinia_lud`, `rajaperf_cuda`, `dual_fft`), junto al F1
   actual. No cambia el modelo, separa "qué tan bien discrimina cuando SÍ
   hay algo que discriminar" de "qué tan bien adivina la identidad de una
   familia entera". Barato, no toca datos.

3. **Ampliar el catálogo con más kernels cercanos al ridge** (más pliegues
   mixtos como los 3 que ya existen). Única palanca que ataca la causa
   raíz. Revisar primero `Nota_Candidatos_GPU_Compute_Bound_20260914.md`
   (sin commitear en el repo al momento de escribir esto) por si ya
   explora candidatos. Requiere clúster.

4. **Recuperar `rodinia_lavamd`/`rodinia_dwt2d`** alargando su duración de
   corrida para que el warmup calibrado (6.4-6.5 s) deje ventanas de
   estado estable suficientes. Ver memoria
   `project-pendiente-alargar-lavamd-dwt2d`. Requiere clúster
   (`paccaA100`).

Descartado explícitamente: cambiar el protocolo `leave-one-familia-out` por
un split que no retenga familias completas (rompería el diseño
experimental de la tesis, que prueba generalización a algoritmos nunca
vistos); reabrir la caza de fases intra-kernel vía ventaneo temporal de
telemetría NVML (ya cerrada en negativo para CPU, ver memoria
`intra-kernel-phase-hunt-negative`, techo F1 macro ~0.5 documentado, causa
física: "varias subrutinas distintas pueden caer del mismo lado del
ridge").

## Idea nueva, pendiente de explorar: perfilado por-lanzamiento con `ncu`

Distinta de la caza de fases ya cerrada: en vez de perfilar UNA vez con
`ncu` y promediar toda la ejecución en una sola OI, aprovechar que `ncu`
ya reporta métricas por cada lanzamiento individual de kernel CUDA. Si un
binario del catálogo invoca internamente varios kernels CUDA distintos (o
el mismo kernel muchas veces con roles distintos), cada lanzamiento podría
tener su propia OI y por tanto su propia etiqueta, en vez de una sola fija
para todo el binario.

Candidato concreto a verificar primero: `rodinia_gaussian` (la eliminación
gaussiana de Rodinia es conocida públicamente por tener dos kernels CUDA
distintos, `Fan1` y `Fan2`, con roles de intensidad aritmética muy
distintos) — **sin verificar todavía contra el código fuente vendido en
este catálogo**, solo es lo que documenta el diseño público de Rodinia.
Justo `rodinia_gaussian` es la familia con la falla más rara del stump
(§3.9.4 del libro), así que sería el primer caso a inspeccionar.

**Importante, para no confundir dos ejes distintos del pipeline** (aclarado
en el chat tras una pregunta legítima): esta idea solo cambia cómo se
calcula la ETIQUETA (ground truth, offline, solo para entrenar/evaluar).
El modelo sigue prediciendo siempre con las 4 features NVML
(`gpu_util_pct`, `gpu_mem_util_pct`, `gpu_power_mw`, `gpu_sm_clock_mhz`),
en entrenamiento y en producción por igual — `ncu` nunca es input del
modelo, es demasiado costoso para correr junto a la carga real. El
beneficio esperado no es "mejor señal de predicción", sino pliegues de
prueba potencialmente menos homogéneos (si de verdad hay fases internas
con OI distinta). El techo de que NVML no distingue ocupación de
intensidad aritmética (ya diagnosticado en `rodinia_gaussian`, §3.9.4)
sigue intacto pase lo que pase con las etiquetas.

Riesgo conocido de antemano (por el negativo ya documentado en
`intra-kernel-phase-hunt-negative`): que ambos kernels/fases caigan del
mismo lado del ridge de todas formas, en cuyo caso no aportaría mezcla
real. No se sabe sin perfilar. Requiere `ncu` en `paccaA100`, así que
sigue bloqueado hasta liberar el nodo.

## Resultado real (2026-09-16): negativo, mismo patrón que el intra-kernel-phase-hunt ya cerrado

Perfilado real con `ncu` (convergió limpio, cambio relativo 0.68% a
launch-count=500) confirma que `rodinia_gaussian` sí invoca dos kernels
CUDA distintos, `Fan1` y `Fan2` (verificado contra el CSV crudo de `ncu`
de este proyecto, no solo el diseño público de Rodinia). Su OI real
difiere genuinamente entre sí:

- `Fan1`: 250 lanzamientos, OI=0.1533 FLOP/byte
- `Fan2`: 250 lanzamientos, OI=0.2838 FLOP/byte

Pero **ambos caen del mismo lado del ridge**: incluso el más alto (Fan2,
0.284) está casi dos órdenes de magnitud por debajo de cualquier ridge
posible en este nodo (3.36-7.28 FLOP/byte). La diferencia entre fases es
real pero demasiado pequeña para cruzar la frontera Roofline -- **mismo
resultado cualitativo que `intra-kernel-phase-hunt-negative`** (variación
estructural real que no alcanza a mezclar clases). Esta idea se cierra
para `rodinia_gaussian`; no se probó en otros kernels multi-lanzamiento
del catálogo por la misma razón de fondo (perfilar por-lanzamiento no
ataca el desbalance si ninguna fase individual cruza el ridge).

Colateral útil: de paso se remidió la OI agregada de `rodinia_gaussian`
con una serie de convergencia completa (10/50/100/500 lanzamientos) en vez
del muestreo de 20 lanzamientos usado originalmente (ARC-110) -- **0.2913
FLOP/byte, converged**, contra el 0.2814 declarado en el catálogo. Mismo
orden de magnitud, no cambia la clasificación; queda como nota de
precisión, no una corrección urgente del catálogo.

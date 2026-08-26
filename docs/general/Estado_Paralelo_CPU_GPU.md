# Estado paralelo CPU / GPU — 2026-08-26

Documento vivo, condensado. Actualizar en el mismo turno en que cambie
algo relevante — no dejar que se desactualice como pasó con
`Estado_Cola_Slurm.md` antes de esta convención.

**No mezclar los dos ejes al leer esto** — son catálogos, umbrales y
modelos independientes.

---

## MODELO CPU

**Catálogo**: 9 kernels originales (Clase B) + 2 confirmados viables.

| viable | resultado |
|---|---|
| `npb_mg` | óptimo real en S3000, EDP/F0=0.9927 (rejilla fina, 638/720 aceptadas) |
| `ptrchase` | α=0.097 (F0–F1, r²=1.000), campaña completa 320/320 |

Los otros 7 originales son compute-bound genuinos — no artefacto de
tamaño: superan la L3 real de `paccaA100` (12 MB) entre 5× y 43×, pero
reutilizan datos intensamente (DGEMM, N-cuerpos, solvers implícitos ADI).
`npb_mg` es la excepción porque su intensidad operacional es baja por
diseño del algoritmo (stencil), no por tamaño.

**En cola en `paccaA100`** (detrás de trabajos ajenos):
- **6579** — Clase C: ¿8× tamaño cruza el umbral en `npb_cg`/`npb_mg`?
- **6575** — tamizaje v2: ~79 kernels RAJAPerf a tamaño correcto (10× LLC
  real = ~120 MB); incluye 12 candidatos de alta confianza (familia
  STREAM, sin reutilización posible por construcción)

**Bloqueado**: GAP Benchmark (`bfs`/`pr`) — binarios listos, sin permiso
de escritura de frecuencia en `pacca01`. Huella de caché a reloj nativo
(`perf`) da resultado incierto (LLC-miss 3.5–10.4%, ni claramente
memory-bound ni claramente compute-bound). Pendiente de tamizar directo
en `paccaA100`.

**El modelo** (piloto LOKO, dataset viejo): pierde contra no hacer nada
(EDP loss 1.0027 vs trivial 1.0010). Causa raíz medida: N efectivo = 8
kernels de entrenamiento, no filas; RMSE 92× el margen disponible. No se
ha reentrenado con `npb_mg`/`ptrchase` — pendiente del tamizaje v2.

---

## MODELO GPU

**Catálogo**: 7 originales + 6 nuevos RAJAPerf-CUDA = 13, OI real medida
por `ncu` (job 6528), checksum verificado.

**Manifiesto final (17 kernels) listo, NO lanzado a propósito** — espera
el resultado de:

**En cola en `paccaA100`**: **6571** — clasificación de cuello de botella
(`ncu` DRAM% vs SM%) sobre los 79 kernels CUDA. Decide si el catálogo
final necesita más candidatos antes de gastar ~10-11h de dataset. (Primer
intento, 6539, falló en silencio por bug propio de parseo — corregido.)

**El modelo** (mismo piloto LOKO, dataset viejo de 6 kernels): pierde
contra no hacer nada, peor que CPU (EDP loss 1.0925 vs trivial 1.0507,
−4.18 pts). Margen real disponible: 5.07 pts. RMSE 5.4× el margen (menos
grave que CPU). Umbral de acción acota la pérdida a cero, no da ganancia.
No se ha reentrenado — pendiente del catálogo final de 6571 y el dataset
de 17 kernels.

---

## En común

- Mismo diagnóstico raíz en los dos ejes: N efectivo pequeño (kernels, no
  filas) es la causa del fracaso del modelo, no un problema de ajuste.
- Mejoras de features identificadas y sin hacer en ninguno: quitar
  `ref_running_ratio` (varianza cero en CPU), enriquecer con
  percentiles/dispersión en vez de solo la media, predecir en espacio log.
- Ningún modelo se ha reentrenado desde los hallazgos de esta sesión —
  ambos esperan a que cierre la ampliación de catálogo respectiva.

## Hallazgos transversales (aplican a ambos ejes o los conectan)

- **La memoria de CPU no se ralentiza con el reloj de núcleo** (medido,
  `uncore_imc_0/clockticks`): el controlador se mantiene en 2.68–2.89 GHz
  mientras el núcleo cae 4×. Lo que cae es el ancho de banda alcanzado
  (−30%) — el núcleo lento no logra saturar una memoria que sí va rápido.
  Límite de carga/catálogo, no de plataforma.
- **GPU solo tiene un reloj de memoria soportado** (1215 MHz, `nvidia-smi
  --query-supported-clocks=mem`) — no hay segundo mando de actuación en
  este acelerador, a diferencia de CPU donde el `uncore` sí es un dominio
  independiente.
- **Overhead de instrumentación caracterizado en CPU**: 1.95% media sobre
  540 pares, estable entre kernels. No cuantificado por separado en GPU.
- **Tamaño de kernel ≠ memory-bound**: exceder la caché es necesario pero
  no suficiente. La intensidad operacional (reutilización de datos, no
  volumen) es lo que decide — mismo principio que llevó a corregir el
  veredicto del tamizaje v1 de CPU y a diseñar v2 con `--memory-touched`.

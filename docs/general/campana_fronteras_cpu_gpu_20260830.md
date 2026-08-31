# Campaña dirigida a fronteras CPU/GPU

**Estado:** preparada, no enviada a Slurm.
**Fecha de diseño:** 2026-08-30.
**Alcance:** extensión posterior al conjunto confirmatorio `*_big_ref_*`; no
modifica ni reutiliza sus nueve tamaños reservados.

## Pregunta

La campaña no intenta aumentar artificialmente la exactitud de un
clasificador. Contrasta si las fronteras de dispositivo que una política por
operación y tamaño infirió del conjunto exploratorio son físicas y estables:

| Operación | Cruce estimado exploratorio | Tamaños nuevos predefinidos |
|---|---:|---:|
| AXPY | N ≈ 208114 | 160000, 208000, 250000 |
| SpMV | N ≈ 2081139 | 1600000, 2080000, 2500000 |
| stencil | N ≈ 3584 | 3328, 3584, 3840 |
| FFT | no monótona entre N=192 y N=512 | repetir 192, 256, 384, 512 |

AXPY, SpMV y stencil cubren ambos lados de su cruce con un punto central. En
FFT no se crean tamaños artificiales: se obtienen seis repeticiones nuevas de
los cuatro tamaños que ya mostraron la anomalía, para estimar si el patrón se
sostiene frente al ruido de la región fría.

## Diseño congelado para esta extensión

- Dos ejes pareados: CPU REF y GPU REF.
- Trece `config_id` por eje; seis repeticiones por combinación: 78 corridas
  CPU y 78 GPU.
- Cada corrida conserva las regiones `cold` y `warm`; se podrán reconstruir
  los tres estados de recurso y el horizonte `K` sin barrido de frecuencias.
- Turbo desactivado, RAPL, uncore, temperatura, telemetría GPU y validación
  de frecuencia por ventana bajo los mismos contratos de campaña vigentes.
- No hay barrido cartesiano de frecuencia. La pregunta es la frontera de
  dispositivo, no el óptimo DVFS fino.

Los nueve tamaños nuevos están en `catalog.yaml`, emitidos de forma
reproducible por `scripts/pacca/gen_dual_boundary_catalog.py`. Sus iteraciones
CPU se calculan por interpolación entre **dos tiempos por despacho medidos**
que los encierran; no se extrapola rendimiento ni se añade un binario nuevo.
Las entradas GPU usan el modelo de dos términos ya usado y validado para la
rejilla dual. Los checksums son los de los binarios CPU/GPU existentes.

## Secuencia de ejecución

1. Esperar la finalización y publicar el resultado de evaluación externa de
   los jobs 6763/6764 con el protocolo confirmatorio ya congelado.
2. Ejecutar preflight normal en paccaA100. Si una nueva duración no produce
   las ventanas o resolución requeridas, la corrida queda registrada y se
   rechaza; no se ajustan iteraciones después de observar EDP.
3. Enviar los dos scripts:

   - `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_cpu_boundary_ref.sbatch`
   - `orchestrator/schemas/scripts/launchers/run_campaign_pacca_dual_gpu_boundary_ref.sbatch`

4. Analizar ambos ejes pareados antes de incorporarlos al entrenamiento final.
   Se reportarán las razones CPU/GPU por tamaño, intervalos entre repeticiones,
   estabilidad de la dirección del cruce y el comportamiento separado de
   `cold` y `warm`.

## Regla de interpretación

Esta campaña puede reforzar o refutar la política de umbrales. No se empleará
para cambiar retrospectivamente el resultado confirmatorio ni para declarar
que ML supera a la baseline por haber elegido tamaños favorables. Si la
frontera resulta estable, la tabla por operación/tamaño sigue siendo la
política de dispositivo principal; el modelo estructurado conserva su papel
de predictor de costos, amortización y, posteriormente, soporte para DVFS.

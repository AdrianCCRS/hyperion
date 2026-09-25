# Protocolo confirmatorio E-A: daemon GPU frente a REF y F1 fijo

**Fecha de declaración:** 24 de septiembre de 2026, antes de someter la campaña.

## Pregunta

En la aplicación E-A ya declarada, ¿el daemon GPU reduce el EDP del nodo frente a REF y frente a mantener F1 durante toda la aplicación?

## Aplicación y brazos

Se reutiliza `composite_gpu_memdom.py --set known`, un ciclo: tres fases GPU `memory_bound` largas (`dual_stencil_gpu_N36864`, `dual_spmv_gpu_N200000000`, `dual_axpy_gpu_N1280000000`) y una fase `compute_bound` (`gpu_cutlass_simt_dgemm_n4096`). Se mantienen `GAP_S=1` y la política GPU congelada: F1 = 1260 MHz en memoria y reloj nativo en cómputo.

Cada brazo se ejecuta cinco veces:

| Brazo | Configuración |
|---|---|
| `base` | REF: reloj GPU nativo, sin daemon. |
| `sombra` | Daemon GPU activo para observar y clasificar, sin escribir frecuencia. |
| `activo_gpu` | Daemon GPU sin daemon CPU: F1 en memoria y nativo en cómputo. |
| `fijo_gpu_f1` | Sin daemon CPU ni GPU; GPU fijada a 1260 MHz durante toda la aplicación. |

El brazo fijo escribe `nvidia-smi -lgc 1260,1260` antes de arrancar la aplicación, registra utilización y reloj SM cada 250 ms y exige al menos tres lecturas con utilización de 10 % o más dentro de 1230–1290 MHz. Libera el candado al finalizar toda celda, también ante salida anómala.

## Orden y ejecución

Se ejecutan 20 celdas en cinco bloques, uno por repetición. Cada bloque contiene los cuatro brazos una vez; su orden se aleatoriza con semilla `20260925` (`ORDER_MODE=balanced_blocks`). La lista exacta queda en `cells.txt` antes de comenzar las mediciones. El nodo es exclusivo y no se usan shells adjuntas para ejecutar trabajo durante una celda.

La salida es `~/hyperion-results/final/fase4_EA_confirmatorio`. El trabajo depende de ninguna cadena de Slurm: se encola después de D (`7682`) y de la premedición F (`7683`) para preservar el orden FIFO ya existente.

## Métrica y análisis fijados

La métrica primaria es el EDP del nodo de cada celda, `(E_CPU + E_GPU) × T`. Se informarán también duración, energía CPU, energía GPU, EDP de GPU sola, decisiones del daemon y registros por fase.

Las comparaciones confirmatorias son bilaterales, `activo_gpu` frente a `base` y `activo_gpu` frente a `fijo_gpu_f1`. Se usarán la razón de medianas, los valores de cada réplica y una prueba exacta de permutaciones bilateral. Los dos valores *p* se ajustarán con Holm para controlar el error familiar a 0,05; una separación completa de cinco contra cinco produce un mínimo bilateral de 0,00794 antes del ajuste, pero no constituye significancia automática. `sombra` cuantifica el costo de observación; su comparación es descriptiva.

## Criterios de validez y exclusión

Una celda solo entra al análisis si `app_rc=0`, `rc_gpu_daemon=0` cuando aplique, `state_ok=1`, todas las fases terminan y NVML/RAPL entregan valores finitos. En `fijo_gpu_f1`, además debe pasar la verificación de candado bajo carga y la restauración inmediata del reloj, ambas registradas en `fixed_gpu_lock.txt`. Se conservarán en el directorio y se reportarán todas las celdas excluidas, su razón y cualquier repetición solicitada para completar los cinco datos válidos por brazo. No se cambiarán kernels, frecuencia, umbral del daemon, métricas, repeticiones ni criterios a partir de los resultados.

# 01 — Nodo Cartagena (`paccaA100`): specs y lo que hay que decidir antes de medir

## Identidad

- Nombre en el clúster: `paccaA100`
- Alias del proyecto: **Cartagena** (también referido como "Pacaca" indistintamente)
- Acceso: SSH + Slurm. Ejemplo de reserva visto en uso:
  ```
  srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --cpus-per-task=32 \
       --gres=gpu:1 --exclusive --pty bash -i
  ```

## `lscpu` (registrado tal cual se recibió)

```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Thread(s) per core:  2
Core(s) per socket:  8
Socket(s):           2
NUMA node(s):        2
Model name:          Intel(R) Xeon(R) Gold 5315Y CPU @ 3.20GHz
CPU max MHz:         3600.0000
CPU min MHz:         800.0000
L1d cache:            48K
L1i cache:            32K
L2 cache:           1280K
L3 cache:          12288K
NUMA node0 CPU(s):   0-7,16-23
NUMA node1 CPU(s):   8-15,24-31
Flags incluyen: avx avx2 avx512f avx512dq avx512cd avx512bw avx512vl
                avx512ifma avx512vbmi avx512_vnni fma ...
```

Total: 2 sockets × 8 cores físicos × 2 hilos SMT = 32 CPUs lógicas, 2 nodos NUMA.

## Por qué este nodo SÍ permite Hardware EBS de VTune (a diferencia de Félix/Westmere)

Este es un Xeon Gold 5315Y, microarquitectura **Ice Lake-SP** (lanzado abril 2021).
VTune soporta oficialmente procesadores Xeon basados en Ice Lake y familias
posteriores para Hardware Event-Based Sampling. El nodo anterior (Westmere, familia
6 modelo 44) estaba muy por debajo de ese corte y por eso VTune no podía hacer EBS
ahí — ver el análisis extenso guardado en el histórico de este proyecto si hace
falta releerlo, pero **no es relevante para el trabajo en este nodo**, solo
contexto de por qué ahora sí se puede confiar en VTune con contadores de hardware.

**Aun así, confirmar con una prueba de humo real antes de construir el pipeline
completo** (ver `PLAN.md` Fase 0) — la documentación dice que debería funcionar, eso
no reemplaza la verificación empírica en este nodo específico.

## Decisión pendiente: dominio de cores (SMT + dos sockets)

Con 8 físicos × 2 sockets × SMT, hay ambigüedad real en cómo correr los kernels. Un
ejemplo de configuración visto en uso para este nodo fue:

```
export OMP_NUM_THREADS=16
export OMP_PLACES=cores        # (recomendado añadir explícitamente, ver abajo)
export OMP_PROC_BIND=spread
```

**Problema:** `OMP_NUM_THREADS=16` sin `OMP_PLACES=cores` es ambiguo — puede acabar
usando SMT en unos cores y dejando otros físicos libres, o puede repartir los 16
hilos cruzando los dos sockets (16 = los 16 físicos de ambos sockets combinados, o
16 lógicos dentro de un solo socket usando SMT — son escenarios muy distintos y el
comando no distingue entre ellos sin `OMP_PLACES`).

**Decisión recomendada (default de este proyecto) — ACTUALIZADA 2026-08-07:**
6 cores físicos (0-5), sin SMT, fijados con `taskset -c 0-5` además de las
variables OMP. Se abandona el default anterior (8 cores, todo el socket 0) al
descubrir que el orquestador principal de Hyperion ya corre campañas reales en
este mismo nodo con `delegated_cpus=0-5` + `collector_cpu=6` + `consumer_cpu=7`
(`orchestrator/schemas/campaign_pacca_ref.yaml`) — usar 8 cores mediría un
dominio distinto y volvería las clasificaciones de VTune no comparables
kernel-por-kernel contra las del orquestador. Ver `context/02_decisiones.md`
D6 (actualizada) para el detalle de la decisión.

```
taskset -c 0-5 <comando>   # o vía run_vtune_pipeline.py --core-range 0-5 (default)
export OMP_NUM_THREADS=6
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

`OMP_PLACES=cores` solo no garantiza cuáles 6 cores se usan — de ahí el
`taskset` explícito, mismo patrón que usa el orquestador
(`campaign_pacca_ref.yaml`, comentario de la sección `cores`).

Si en algún momento se decide medir con los dos sockets o con SMT activo, debe ser
una decisión explícita y documentada como una caracterización distinta — no algo que
quede implícito en una variable de entorno ambigua. VTune reporta `DRAM Bound` y
tráfico NUMA por socket; mezclar dominios sin decirlo invalida la comparación entre
kernels.

## Confirmación de exclusividad para que las métricas de memoria signifiquen algo

```bash
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE
```

Los cores usados por el pipeline deben caer todos en el mismo NODE/SOCKET. La
reserva de Slurm debe ser `--exclusive` — sin eso, `DRAM Bound` y `Memory Bound`
de VTune pueden reflejar tráfico de otro job compartiendo el socket.

## Sobre LIKWID en este nodo

**No se usa.** No hay permisos todavía para instalarlo ni para el acceso a
contadores que necesitaría (`perf_event_paranoid`, o el daemon setuid). No es un
bloqueo para este trabajo — el pipeline de VTune no depende de LIKWID en absoluto.
Si en el futuro se consiguen permisos, es una extensión posible, no un requisito.

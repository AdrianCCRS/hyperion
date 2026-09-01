# 03 — Notas por kernel (relevantes para interpretar lo que VTune reporte)

## EP — Embarrassingly Parallel

- Naturaleza: generación de pares gaussianos (método polar de Marsaglia), núcleo de
  Monte Carlo. Trabajo dominante: generador congruencial de 64 bits, `sqrt`, `log`.
- **Riesgo de medición:** el `DP GFLOPS` que reporta VTune típicamente proviene de
  contar operaciones SSE/AVX de suma y multiplicación empaquetadas/escalares. La
  unidad de división/raíz cuadrada (usada intensivamente por EP) puede no estar
  incluida en ese mismo contador, dependiendo de qué eventos arme VTune para esa
  métrica en Ice Lake. Si es así, EP puede aparecer con GFLOPS bajos que no reflejan
  su verdadera carga de cómputo.
- **Qué hacer:** si `hpc-performance` expone algo equivalente a operaciones de
  división/latencia de unidad de división (revisar qué eventos hardware expone
  `-report hw-events` para este kernel), registrarlo aparte como diagnóstico. Si
  EP sale clasificado como `memory_bound` o `ambiguous`, marcarlo en el reporte
  como "requiere revisión manual — posible subconteo de FLOPs", no aceptarlo sin
  más como resultado válido.
- Working set: diminuto y prácticamente constante entre clases — no crece
  significativamente con A/B/C/D.

## IS — Integer Sort

- Naturaleza: ordenamiento de claves enteras por conteo/cubetas. **No tiene FLOPs.**
- Cualquier métrica de VTune basada en `DP GFLOPS`/`SP GFLOPS` para IS será cero o
  cercana a cero por construcción — no es evidencia de nada, es un resultado
  degenerado. Clasificar IS en un esquema memory/compute-bound basado en FLOPs es un
  error categorial, no una imprecisión de medición.
- **Qué hacer si se retoma IS en este nodo:** excluirlo del cálculo de clasificación
  basado en GFLOPS, o usar una métrica alternativa (ej. instrucciones retiradas por
  byte movido) documentada aparte. No forzar la etiqueta `memory_bound` solo porque
  el numerador de FLOPs sea cero.

## CG — Conjugate Gradient

- SpMV + productos punto sobre una matriz dispersa **aleatoria** (no proviene de una
  malla física — menos localidad que un SpMV real).
- Buen candidato como referencia del extremo memory/latency-bound si se necesita un
  segundo punto de calibración además de STREAM, porque a diferencia de STREAM sí
  tiene FLOPs reales y reducciones con barrera — más parecido a una aplicación real.

## MG — Multigrid

- Recorre una jerarquía de mallas (V-cycles) dentro de una misma corrida. Los
  niveles finos no caben en la LLC (memory-bound); los niveles gruesos sí
  (intermedio/latency-bound). **Un solo binario produce fases de ambos regímenes.**
- Si el pipeline solo mide un valor agregado por corrida completa (que es lo que
  hace `hpc-performance` estándar), esa mezcla de fases queda promediada y puede
  salir como `ambiguous` sin que eso sea un error — es el comportamiento esperado
  de este kernel en particular. Documentarlo así en `classification_justification`
  cuando ocurra, no tratarlo como una corrida sospechosa.

## FT — Fast Fourier Transform

- Alterna FFT locales (buena localidad) con transposiciones globales (tráfico
  masivo). Mismo fenómeno que MG: mezcla de fases dentro de una corrida.

## LU — SSOR Gauss-Seidel

- Paralelismo por *wavefront* con dependencias fuertes y sincronización explícita
  entre hilos. Puede aparecer con métricas intermedias no porque esté balanceado
  entre memoria y cómputo, sino porque el limitante real es la sincronización — algo
  que ni VTune ni un Roofline clásico capturan directamente como una tercera
  categoría. Si sale `ambiguous`, vale la pena anotar esta posible causa en la
  justificación en vez de dejarlo como "métricas mixtas" sin más explicación.

## BT / SP — candidatos a ancla compute-bound (ver decisión D3)

- Solvers estructurados con bloques densos, aritmética SSE/AVX estándar (suma y
  multiplicación, sin división/transcendentales pesados como EP). Buenos
  candidatos si DGEMM/OpenBLAS no está disponible en el nodo como kernel ancla de
  calibración.

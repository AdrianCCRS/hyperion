# Pendiente: cerrar la validación del fix de `scale_pos_weight` (CPU)

## Contexto

`cb84d33` corrigió `scale_pos_weight` de XGBoost (recalculado por pliegue
externo, antes congelado). Verificado leyendo `model_specs.py`: de los 7
modelos comparados, **solo XGBoost** usa ese parámetro — los otros 6
(`mayoritaria`, `arbol_prof1`, `regresion_log`, `arbol_prof6`,
`random_forest`, `extra_trees`) usan `class_weight="balanced"` de sklearn
(sin bug) o no se ponderan (los dos fijos).

`7355` (`hyp_cpu_optuna`, lanzado en `pacca03` antes del fix, corriendo desde
2026-09-15 20:06) sigue vivo con el código VIEJO. Como el fix no toca a los
otros 6 modelos, **no hace falta matarlo ni repetir los 7**: sus resultados
para esos 6 siguen siendo válidos cuando termine.

Ya hay una corrida preliminar real con el fix (`7387`, `pacca07`,
`n-trials=8`, submuestreada) que confirma que el fix funciona: XGBoost con
fix da F1 macro=0.629, F1 compute_bound=0.450, ventaja sobre baseline
+0.186 (`~/hyperion-results/preliminar_cpu_20260916/xgboost.metadata.json`
en pacca). Pero es liviana (8 trials, `per-run-sample=200`), no reemplaza
la corrida oficial de 30 trials sin submuestreo.

## Qué se agregó (commit `bb2cbf7`, pusheado a `origin/main`)

`fase2_clasificador/training/train_phase.py` ahora acepta
`--only-models <lista-separada-por-coma>` para restringir qué modelos (de
los 7) se entrenan/evalúan en una corrida, sin tocar el comportamiento por
defecto (sin el flag, corren los 7 igual que siempre). Filtra
`fixed_models`, `tunable_names` y el `tunable` recalculado por pliegue.

## Pasos para cerrar esto

No hay dependencia real entre `7355` y esto: son corridas de Fase 2
independientes sobre el mismo dataset ya medido, así que se puede lanzar
en paralelo en un nodo libre distinto (`pacca05`/`pacca07`), sin esperar a
que `7355` cierre.

1. `git pull origin main` en pacca para traer `bb2cbf7`.
2. Lanzar SOLO XGBoost con el fix, 30 trials reales, sin submuestreo
   (mismos args que la oficial, agregando `--only-models xgboost`):

   ```bash
   ssh hpc-unicartagena
   ssh pacca
   cd ~/hyperion && git pull origin main -q
   source ~/hyperion-venv/bin/activate
   export PYTHONPATH=~/hyperion

   KERNELS="npb_bt,npb_mg,npb_cg,npb_sp,npb_ft,npb_lu,dgemm_n2048,rodinia_lavamd_omp,rajaperf_polybench_3mm_omp,cpu_rajaperf_stream_mul,cpu_rajaperf_stream_triad,cpu_rajaperf_stream_add,cpu_rajaperf_lcals_first_sum,cpu_rajaperf_lcals_tridiag_elim,cpu_rajaperf_polybench_jacobi_1d,cpu_rajaperf_polybench_fdtd_2d,cpu_rajaperf_basic_daxpy,cpu_rajaperf_basic_init3,cpu_lulesh,cpu_hpcg,cpu_gap_pr,cpu_cholmod,phasic_p010,phasic_p100,phasic_p1000,dual_gemm_cpu_N2048,dual_fft_cpu_N128,dual_fft_cpu_N1024,dual_fft_cpu_N4096,dual_axpy_cpu_N100000,dual_axpy_cpu_N3162278,dual_axpy_cpu_N10000000,dual_stencil_cpu_N256,dual_stencil_cpu_N512,dual_stencil_cpu_N1024,dual_stencil_cpu_N1536,dual_cholesky_cpu_N256,dual_cholesky_cpu_N1024,dual_cholesky_cpu_N2048,dual_spmv_cpu_N100000,dual_spmv_cpu_N1000000,hpccg_cpu_N46,dual_fft_cpu_N472"

   srun -p normal --ntasks=1 --mem=32G --job-name=hyp_cpu_xgb_only \
     python3 fase2_clasificador/training/train_phase.py \
       --campaign-dir "$HOME/hyperion-results/final/campaigns/cpu" \
       --campaign-id pacca_cpu_final_20260913 \
       --kernels "$KERNELS" \
       --levels REF,F0,F1,F2,F3,F4,F5,F6,F7,F8 \
       --n-trials 30 \
       --only-models xgboost \
       --output-dir "$HOME/hyperion-results/xgboost_only_20260916"
   ```

   (recordar prefijo `hyp_`, nunca `--time`, nunca cancelar jobs ajenos).

3. Cuando ambos ya hayan terminado (`7355` y esta corrida), fusionar el
   resultado: tomar la entrada `xgboost` de
   `xgboost_only_20260916/xgboost.metadata.json` (bloque
   `all_models_compared.xgboost` + el modelo serializado si termina siendo
   el elegido) y reemplazar/insertar esa entrada en el `metadata.json`
   oficial que dejó `7355`. Volver a correr `select_best_model()` a mano
   (o inspeccionar el score) para confirmar si el modelo elegido cambia.
4. Actualizar el capítulo de resultados del libro con el número final,
   citando que XGBoost se revalidó por separado tras el fix, sin repetir
   los otros 6 (nota de higiene metodológica, no un bug).

## Nota

No crear una campaña ni tocar Fase 1 (telemetría) para esto — es puro
reentrenamiento de Fase 2 sobre datos ya medidos, mismo dataset e
hiperparámetros que la oficial, solo con el alcance de modelos reducido.

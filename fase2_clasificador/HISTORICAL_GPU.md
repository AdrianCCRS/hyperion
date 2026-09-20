# Reanálisis de campañas GPU históricas (sin CUPTI)

Esta ruta sirve exclusivamente para rescatar campañas ya completadas que solo
contienen ventanas NVML y una etiqueta Roofline estática por corrida. No
produce fases nuevas ni pretende sustituir el contrato CUPTI por ventana.

Primero se construye una observación independiente por corrida. Por defecto se
acepta solo `quality_status=gpu_telemetry`; warmup, filas inválidas y corridas
rechazadas quedan fuera antes de cualquier agregado.

```bash
python3 -m fase2_clasificador.analysis.historical_gpu_runs \
  local_datasets/final_campaigns_20260821/gpu_windows.csv.gz \
  --output tmp/historical_gpu_runs.csv
```

Después se comparan agregados robustos y derivados físicos: mediana, media
ponderada por tiempo, IQR/p90/media recortada, potencia normalizada por reloj,
y las razones potencia/utilización y memoria/utilización. También se puede
incluir la distribución completa de cada señal. Las razones son candidatas
explícitas, no una decisión asumida de antemano: se conservan incluso si no
ganan el ranking exploratorio.
La evaluación siempre retiene familias completas, tanto en LOFO como en folds
mixtos; los resultados se escriben completos, incluido el peor fold y la matriz
de confusión.

```bash
python3 -m fase2_clasificador.analysis.evaluate_historical_gpu_runs \
  --dataset tmp/historical_gpu_runs.csv \
  --output-dir tmp/historical_gpu_evaluation \
  --nested-selection --frequency-sensitivity
```

Las campañas extra solo deben pasarse como entradas adicionales al primer
comando tras comprobar que usan el mismo esquema, controles de calidad y
provenencia de etiqueta. El reporte resultante no autoriza mezclar filas por
ventana de estas campañas con `training_gpu_phases.csv` de CUPTI.

El constructor conserva `source_campaign_id` y rechaza que un mismo `run_id`
describa campañas, kernels o frecuencias distintas. Antes de una fusión final,
el reporte debe listar todas las campañas previstas y el balance de familias
debe verificarse explícitamente; la herramienta no reetiqueta ni rebalancea
datos.

`--nested-selection` elige la representación y el modelo únicamente dentro de
cada entrenamiento externo por familias; su F1 no es el ranking exploratorio.
`--frequency-sensitivity` deja fuera cada nivel GPU como diagnóstico adicional.
No sustituye LOFO/folds mixtos, porque las mismas familias siguen presentes en
ambos lados de ese control.

Para fusionar campañas modernas por corrida (nunca ventanas CUPTI) se usa:

```bash
python3 -m fase2_clasificador.analysis.merge_historical_gpu_phase_runs \
  path/a/training_gpu_phases.csv path/a/otra/training_gpu_phases.csv \
  --output tmp/historical_gpu_merged.csv
```

La ponderación equitativa por familia evita que una familia con muchas corridas
domine el ajuste. Se activa con `--family-balanced-training`. La búsqueda
regularizada se ejecuta de forma anidada para que C, penalización y la
representación se elijan sin observar el fold externo:

```bash
python3 -m fase2_clasificador.analysis.tune_historical_gpu_logistic \
  --dataset tmp/historical_gpu_merged.csv \
  --exclude-family minibude_cuda_bm1 \
  --output tmp/historical_gpu_logistic_tuned.json
```

Por último, la evaluación selectiva no mejora artificialmente la métrica de
cobertura total. Emite `revisar` cuando la confianza es menor al umbral elegido
en el entrenamiento interno y siempre reporta conjuntamente cobertura, F1 de
los casos aceptados y su peor fold:

```bash
python3 -m fase2_clasificador.analysis.evaluate_historical_gpu_selective \
  --dataset tmp/historical_gpu_merged.csv \
  --exclude-family minibude_cuda_bm1 \
  --output tmp/historical_gpu_selective.json
```

## Dinámica temporal NVML

Los agregados robustos pierden el orden de las muestras. Si se conservan los
`samples.csv`, se puede recortar de forma no supervisada el intervalo entre la
primera y última actividad GPU y derivar actividad, ráfagas, autocorrelación,
cambios y correlaciones SM–memoria–potencia. El recorte no consulta etiqueta,
OI, ridge ni identidad del kernel.

```bash
python3 -m fase2_clasificador.analysis.extract_historical_gpu_temporal \
  --raw-root path/a/campaigns \
  --metadata tmp/historical_gpu_merged.csv \
  --exclude-family minibude_cuda_bm1 \
  --output tmp/historical_gpu_temporal.csv
```

La representación `temporal_core` es deliberadamente pequeña: conserva las
cuatro medianas NVML y añade fracciones de actividad, ráfagas, cambios medios y
correlaciones. Debe evaluarse con familia completa fuera. La media por fold y
el peor fold son las cifras primarias; cada F1 macro fuerza explícitamente las
dos clases, incluso si la familia retenida contiene una sola. Un F1 calculado
sobre todas las predicciones concatenadas se reporta solo como diagnóstico
secundario.

## Validación externa entre campañas

Una campaña que no intervino en ningún ajuste debe mantenerse sellada: se usa
una vez para contrastar el candidato congelado, nunca para elegir variantes,
árboles ni umbrales. El evaluador informa F1 macro agrupado, matriz de confusión
y exactitud media por familia; se debe comparar también contra la regla
constante de clase mayoritaria, especialmente si el conjunto externo está
desbalanceado.

```bash
python3 -m fase2_clasificador.analysis.evaluate_historical_gpu_external \
  --train tmp/historical_gpu_temporal.csv \
  --test tmp/historical_gpu_aug_temporal.csv \
  --variant median_iqr --model regresion_log \
  --exclude-test-family dgemm \
  --exclude-test-family rodinia_gaussian \
  --exclude-test-family rodinia_heartwall \
  --exclude-test-family rodinia_lud \
  --output tmp/historical_gpu_external.json
```

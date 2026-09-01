# Campañas de barrido (ARC-89)

Las corridas reales de estos barridos no se guardan en este repositorio
(los datos crudos de telemetría son demasiado grandes) -- viven en
`paccaA100`, bajo un árbol de resultados **separado** de las campañas
reales de construcción del catálogo:

```
~/hyperion-results/sweeps/          <- barridos de este informe (ARC-89)
    interval_ns/
    gpu_interval_ns/
    repetitions/
    ncu_launch_count/
    calibration_repeats/

~/hyperion-results/campaigns/       <- campañas reales del dataset (Fase 1), sin tocar
```

Los scripts que generaron cada barrido están en `docs/justifications/scripts/`,
y los resúmenes ya extraídos (CSV) en `docs/justifications/data/`. El
reporte completo con la interpretación de cada resultado está en
`docs/justifications/report/main.pdf` (fuente en `main.tex`).

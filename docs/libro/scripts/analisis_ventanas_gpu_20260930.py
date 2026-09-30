"""Revisión C2: ¿las entradas que el agente calcula sobre su ventana de 3 s se parecen a las agregadas por corrida con las
que se entrenó el random forest de GPU?

Entradas:
  docs/libro/datos/fase4_20260930/ventanas_agente_gpu.csv   decisiones del agente de GPU en la Fase 4 (jobs 7696 y 7789),
      atribuidas a la fase de la aplicación compuesta que las contiene, con las tres variables de la ventana
  docs/libro/datos/gpu_calidad_20260922/historical_gpu_relaxed020_20260922.csv   corridas de entrenamiento (483)

Para cada kernel visto en el entrenamiento que aparece en las compuestas: mediana de cada variable en las corridas REF,
rango sobre todos los niveles, mediana y p10-p90 en las ventanas, fracción de ventanas fuera del rango de entrenamiento,
decisiones, abstenciones y confianza mínima.
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "datos"
PARES = (("gpu_util_pct_median", "gpu_util_med"), ("gpu_mem_util_pct_median", "mem_util_med"),
         ("gpu_mem_util_pct_std", "mem_util_std"))

w = pd.read_csv(D / "fase4_20260930" / "ventanas_agente_gpu.csv")
t = pd.read_csv(D / "gpu_calidad_20260922" / "historical_gpu_relaxed020_20260922.csv")
for k in sorted(set(w.kernel) & set(t.kernel_ref)):
    tr, ref, ww = t[t.kernel_ref == k], t[(t.kernel_ref == k) & (t.gpu_freq_level_id == "REF")], w[w.kernel == k]
    dec = ww[ww.label != "revisar"]
    print(f"\n{k} ({ww.truth.iloc[0]}): corridas {len(tr)} (REF {len(ref)}); ventanas {len(ww)}, abstenciones "
          f"{(ww.label == 'revisar').sum()}, aciertos {(dec.label == dec.truth).sum()}/{len(dec)}, "
          f"confianza mínima de lo decidido {dec.confidence.min() if len(dec) else float('nan'):.2f}")
    for a, b in PARES:
        fuera = ((ww[b] < tr[a].min()) | (ww[b] > tr[a].max())).mean()
        print(f"  {b:13s} REF {ref[a].median():6.1f}  rango [{tr[a].min():.1f}, {tr[a].max():.1f}]  |  ventanas "
              f"{ww[b].median():6.1f} (p10-p90 {ww[b].quantile(.1):.1f}-{ww[b].quantile(.9):.1f})  fuera de rango {fuera:.0%}")

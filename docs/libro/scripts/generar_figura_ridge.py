#!/usr/bin/env python3
"""Genera figura del ridge Roofline en función del nivel de frecuencia."""

import matplotlib.pyplot as plt
import numpy as np

# Datos del ridge por nivel de frecuencia
niveles = ['REF', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
ridge_valores = [8.69, 8.63, 8.20, 7.31, 6.58, 5.74, 5.03, 4.33, 3.64, 2.92]

# Crear figura
fig, ax = plt.subplots(figsize=(8, 5))

# Graficar
x = np.arange(len(niveles))
ax.plot(x, ridge_valores, marker='o', linewidth=2.5, markersize=7, color='#2E86AB')

# Configurar eje X
ax.set_xticks(x)
ax.set_xticklabels(niveles)
ax.set_xlabel('Nivel de frecuencia', fontsize=11)

# Configurar eje Y
ax.set_ylabel('Ridge Roofline (FLOP/byte)', fontsize=11)

# Agregar grid
ax.grid(True, alpha=0.3, linestyle='--')

# Ajustar layout
plt.tight_layout()

# Guardar figura
output_path = '/home/adrianccrs/Documents/Dev/TG/hyperion/docs/libro/figuras/fig_ridge_por_frecuencia_20260920.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figura guardada en: {output_path}")

plt.close()

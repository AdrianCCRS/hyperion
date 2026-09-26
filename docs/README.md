# Documentación del proyecto

| Carpeta | Contenido |
|---|---|
| `libro/` | Documento de tesis en LaTeX (`main.tex`, capítulos en `secciones/`), figuras, scripts que las generan y `datos/` con los resultados procesados que las respaldan. Ver `libro/datos/README.md` |
| `planeacion/` | Planes por fase, protocolos experimentales (E-A confirmatorio, CloverLeaf), notas de trabajo y cierres. El plan vigente y el seguimiento de cambios están en la raíz del repositorio: `Plan_Detallado_Realineacion_Hyperion.md` y `Seguimiento_Cambios_Plan_Director.md` |
| `general/` | Informes de calidad de modelos |

Compilar el libro: `cd docs/libro && latexmk -pdf main.tex`.

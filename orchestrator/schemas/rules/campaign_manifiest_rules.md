# Validation rules for manifiest to apply on manifiest.py

- Todo kernel referenciado, debe existir en `catalog_path` -> si no, rechazar el manifiest antes de tocar el nodo

- La sección `calibration` debe contener **AL MENOS UN** kernel `role=calibration` con `eports_bandwidth_stdout` y al menos uno con `reports_flops_stdout` -> si falta alguno, no se puede calcular I_ridge, entonces rechazar.

- Ningún kernel con `role=calibration` puede aparecer en la sección kernels (dataset), ni viceversa -> es un error de manifest, no una advertencia.


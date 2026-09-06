# Config local de latexmk para este libro.
# No cambia el backend bibliográfico (sigue siendo BibTeX vía \bibliography{main})
# ni el motor (pdflatex, forzado por el flag -pdf de la propia extensión/recipe).
#
# Por qué existe: con ~150 citas y muchas figuras/labels, latexmk necesita más
# de una ronda de pdflatex+bibtex+pdflatex para converger. En este entorno,
# pdflatex devuelve código de salida distinto de cero en pasadas intermedias
# que solo están "pendientes de un rerun" (undefined references / "Label(s)
# may have changed"), y latexmk por defecto se detiene ahí en vez de seguir
# reintentando. Esto sube el límite de reintentos y le dice a latexmk que siga
# adelante pese a ese código de salida intermedio, en vez de abortar.
$max_repeat = 10;
$force_mode = 1;

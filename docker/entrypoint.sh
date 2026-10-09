#!/bin/sh
# Punto de entrada de la imagen.
#   docker run cfl-practicas comparar --seeds 42 43   -> python /app/cfl.py comparar ...
#   docker run cfl-practicas python domainnet.py       -> se ejecuta tal cual
set -e

# Si se monta otra copia del código (modo dev, o -v ...:/workspace -w /workspace),
# sus módulos tienen prioridad sobre los de la imagen.
if [ -f "$(pwd)/fl_core.py" ]; then
    export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
fi

CFL=/app/cfl.py
if [ -f "$(pwd)/cfl.py" ]; then
    CFL="$(pwd)/cfl.py"
fi

# Alias de los lanzadores antiguos.
case "${1:-}" in
    test)     shift; set -- entorno "$@" ;;
    unittest) shift; set -- tests "$@" ;;
esac

case "${1:-ayuda}" in
    info|entorno|tests|rapido|comparar|semillas|plan)
        exec python "$CFL" "$@" ;;
    ayuda|help|-h|--help)
        exec python "$CFL" --help ;;
    *)
        exec "$@" ;;
esac

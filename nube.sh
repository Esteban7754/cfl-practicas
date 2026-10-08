#!/usr/bin/env bash
# Experimento completo en una máquina Linux con GPU NVIDIA (p. ej. un Pod de RunPod).
#
#   git clone https://github.com/Esteban7754/cfl-practicas.git && cd cfl-practicas
#   nohup bash nube.sh > nube.log 2>&1 &        # sigue aunque cierres el navegador
#   tail -f nube.log                            # ver el progreso
#
# Instala las mismas versiones fijadas que la imagen Docker (requirements.lock), pero con
# PyTorch para CUDA, comprueba la GPU, ejecuta las tres variantes de la comparación y deja
# todo en resultados_nube.tar.gz para descargarlo.
#
# Variables opcionales: SEEDS="42 43 44"  BUFFERS="0 5 10 20"  VARIANTES="normal balanced lwf"
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-"42 43 44"}
BUFFERS=${BUFFERS:-"0 5 10 20"}
VARIANTES=${VARIANTES:-"normal balanced lwf"}

echo "=== Entorno Python 3.13 con las versiones fijadas (PyTorch CUDA)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv -p 3.13 .venv
# Igual que TORCH_VARIANT=cuda en el Dockerfile: mismas versiones, sin el sufijo +cpu.
sed -e '/^--extra-index-url/d' -e 's/+cpu//' requirements.lock > /tmp/requirements-cuda.txt
uv pip install -p .venv/bin/python -r /tmp/requirements-cuda.txt
PY=.venv/bin/python

export MPLBACKEND=Agg PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # algoritmos deterministas en CUDA
export HF_HOME=${HF_HOME:-$PWD/.hf-cache}

echo "=== Comprobación de la GPU"
$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch no ve ninguna GPU: revisa el Pod o el controlador"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
EOF
$PY cfl.py info

echo "=== Tests"
$PY -m unittest -q

for v in $VARIANTES; do
    case "$v" in
        normal)   extra=() ;;
        balanced) extra=(--replay-mix balanced) ;;
        lwf)      extra=(--replay-mix balanced --distill-weight 1) ;;
        *) echo "Variante desconocida: $v" >&2; exit 1 ;;
    esac
    echo "=== Variante $v ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    $PY cfl.py comparar --seeds $SEEDS --buffers $BUFFERS "${extra[@]}"
done

tar -czf resultados_nube.tar.gz --exclude='*.pt' resultados nube.log 2>/dev/null || \
    tar -czf resultados_nube.tar.gz --exclude='*.pt' resultados
echo "=== TERMINADO ($(date '+%F %T')). Descarga resultados_nube.tar.gz y apaga el Pod."

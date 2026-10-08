# Imagen reproducible de los experimentos: dependencias fijadas, CIFAR-100 incluido
# (revisión fijada, funciona sin Internet) y el código del proyecto dentro.
#
#   docker build -t cfl-practicas:cpu .                                    # CPU
#   docker build -t cfl-practicas:gpu --build-arg TORCH_VARIANT=cuda .     # GPU NVIDIA
#   docker build -t cfl-practicas:cpu --build-arg PRELOAD_DATA=0 .         # sin datos (más ligera)
#   --build-arg BASE_IMAGE=... --build-arg TORCH_VARIANT=system             # usa el PyTorch de la imagen base
#                                                                            # (p. ej. nvcr.io/nvidia/pytorch)
#
# Lo normal es no llamar a docker directamente: .\run.ps1 o run.bat lo hacen por ti.

ARG BASE_IMAGE=python:3.13-slim@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b

# ---------------------------------------------------------------- dependencias
FROM ${BASE_IMAGE} AS deps
ARG TORCH_VARIANT=cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg
COPY requirements.lock /tmp/requirements.lock
# La caché de pip (BuildKit) hace que reconstruir tras cambiar el código no vuelva a descargar nada.
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$TORCH_VARIANT" in \
        cpu)  cp /tmp/requirements.lock /tmp/requirements.txt ;; \
        cuda) sed -e '/^--extra-index-url/d' -e 's/+cpu//' /tmp/requirements.lock > /tmp/requirements.txt ;; \
        system) grep -vE '^(--extra-index-url|torch==|torchvision==)' /tmp/requirements.lock > /tmp/requirements.txt ;; \
        *)    echo "TORCH_VARIANT debe ser cpu, cuda o system" >&2; exit 1 ;; \
    esac \
    && python -m pip install -r /tmp/requirements.txt \
    && if [ "$TORCH_VARIANT" = "system" ]; then \
           python -c "import torch, torchvision; print('PyTorch de la imagen base:', torch.__version__)"; \
           python -m pip check || echo "AVISO: pip check con el PyTorch de la imagen base (normal en imágenes NGC)"; \
       else python -m pip check; fi

# ---------------------------------------------------------------- datos
FROM deps AS data
ARG PRELOAD_DATA=1
ENV HF_HOME=/opt/hf-cache
COPY docker/descargar_datos.py /tmp/descargar_datos.py
RUN mkdir -p /opt/hf-cache \
    && if [ "$PRELOAD_DATA" = "1" ]; then python /tmp/descargar_datos.py; fi

# ---------------------------------------------------------------- imagen final
FROM deps AS final
ARG TORCH_VARIANT=cpu
ARG PRELOAD_DATA=1
ARG CFL_REVISION=desconocida
ARG BUILD_DATE=desconocida
LABEL org.opencontainers.image.title="cfl-practicas" \
      org.opencontainers.image.description="Aprendizaje federado continuo con replay buffer" \
      org.opencontainers.image.source="https://github.com/Esteban7754/cfl-practicas" \
      org.opencontainers.image.revision="${CFL_REVISION}" \
      org.opencontainers.image.created="${BUILD_DATE}"
ENV HF_HOME=/opt/hf-cache \
    PYTHONPATH=/app \
    CFL_IMAGE_REVISION=${CFL_REVISION} \
    CFL_IMAGE_BUILD_DATE=${BUILD_DATE} \
    CFL_IMAGE_VARIANT=${TORCH_VARIANT} \
    CFL_DATA_PRELOADED=${PRELOAD_DATA}
COPY --from=data /opt/hf-cache /opt/hf-cache
WORKDIR /app
COPY . /app
# Normaliza finales de línea por si el código viene de un checkout de Windows.
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/resultados /app/data \
    && python -m compileall -q /app
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["ayuda"]

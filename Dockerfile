FROM python:3.13-slim@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b

# PYTHONPATH permite ejecutar los scripts de legacy/ y resultados/ importando
# los módulos comunes de la raíz (fl_core, experiment_common...).
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.lock

CMD ["python"]

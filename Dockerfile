FROM ollama/ollama:latest AS ollama-init
COPY scripts/init-models.sh /usr/local/bin/init-models.sh
ENTRYPOINT ["/bin/sh", "/usr/local/bin/init-models.sh"]

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts/smoke_ollama.py ./scripts/smoke_ollama.py
RUN useradd --create-home agent && mkdir -p /data/documents && chown -R agent:agent /data
USER agent
EXPOSE 8000
CMD ["python", "-m", "app.bootstrap"]

FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
USER agent
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp=/tmp/pytest"]

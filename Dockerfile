FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAMPUS_KNOWLEDGE_DIR=/data/knowledge \
    CAMPUS_VECTOR_CACHE_DIR=/data/vector_cache \
    CAMPUS_SERVER_HOST=0.0.0.0 \
    CAMPUS_SERVER_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY campus_agent ./campus_agent
RUN pip install --no-cache-dir . \
    && addgroup --system campus-agent \
    && adduser --system --ingroup campus-agent campus-agent \
    && mkdir -p /data/knowledge /data/vector_cache \
    && chown -R campus-agent:campus-agent /data

USER campus-agent
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2).read()"]

CMD ["campus-agent-server"]

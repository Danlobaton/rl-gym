FROM python:3.13-slim

WORKDIR /app

# Install just the env-side deps (fastapi + uvicorn). Coordinator/dev extras
# stay out — the env never calls the LLM.
COPY pyproject.toml .
COPY sregym/ ./sregym/
RUN pip install --no-cache-dir .

# Drop root for defense in depth. /tmp is world-writable so /tmp/sregym/ works.
RUN useradd --create-home --uid 10000 app && chown -R app:app /app
USER app

EXPOSE 8000

# Tell orchestrators when uvicorn is actually serving. Uses urllib so we don't
# need curl (slim doesn't ship it).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)"

# 1 worker = 1 episode at a time. Scale by running multiple containers, each
# with its own /tmp/sregym/ and subprocess fanout.
CMD ["uvicorn", "sregym.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

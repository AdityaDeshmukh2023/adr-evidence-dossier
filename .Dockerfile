FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex
ARG WITH_OCR=1
ARG WITH_ML=0
COPY requirements-linux.lock requirements-linux-ocr.lock requirements-ml.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN if [ "$WITH_OCR" = "1" ]; then pip install --no-cache-dir -r requirements-linux-ocr.lock; else pip install --no-cache-dir -r requirements-linux.lock; fi
RUN if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt && pip check; fi
COPY . .
RUN useradd --create-home --uid 10001 reviewer && mkdir -p /app/.cache/paddlex /app/artifacts && chown -R reviewer:reviewer /app
USER reviewer
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "llm.py", "--server.address=0.0.0.0", "--server.port=8501"]

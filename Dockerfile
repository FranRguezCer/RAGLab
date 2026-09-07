FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
RUN mkdir -p src/raglab && touch src/raglab/__init__.py \
    && pip install --no-cache-dir ".[generation,retrieval,tokenizers]"

COPY src ./src
COPY data ./data
COPY scripts ./scripts
RUN mkdir -p artifacts \
    && pip install --no-cache-dir --no-deps . \
    && chown -R 65532:65532 /app
USER 65532:65532
EXPOSE 8000
CMD ["uvicorn", "raglab.demo_cli:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

FROM python:3.12-slim AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Preserve the application lock; replace only the GPU torch distribution on CPU hosts.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate --no-header > /tmp/locked.txt \
    && python -c 'from pathlib import Path; p=Path("/tmp/locked.txt"); p.write_text("\n".join(s for s in p.read_text().splitlines() if not s.startswith(("torch==", "triton==", "nvidia-", "cuda-"))))' \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python --no-cache --no-deps -r /tmp/locked.txt \
    && uv pip install --python /app/.venv/bin/python --no-cache --no-deps --torch-backend cpu 'torch==2.15.0'
COPY src ./src
RUN uv pip install --python /app/.venv/bin/python --no-cache --no-deps .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends util-linux libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home docsqa
WORKDIR /app
COPY --from=dependencies /app/.venv /app/.venv
COPY scripts ./scripts
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER docsqa
CMD ["sh", "scripts/container_start.sh"]

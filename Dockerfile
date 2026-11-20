FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends util-linux libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"
CMD ["sh", "scripts/container_start.sh"]

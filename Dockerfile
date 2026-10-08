FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY mcp_server ./mcp_server

RUN uv sync --locked --no-dev \
    && useradd --system --uid 10001 --user-group --create-home --home-dir /home/kojutsu kojutsu \
    && install -d -o kojutsu -g kojutsu /data \
    && chown -R kojutsu:kojutsu /opt/venv

ENV HOME=/home/kojutsu \
    TANSEKI_OUTBOX_PATH=/data/tanseki-outbox.db \
    KOJUTSU_REGISTRY_PATH=/data/registry.db

USER kojutsu

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/webhook/health', timeout=3)"

CMD ["kojutsu", "serve", "--host", "0.0.0.0", "--port", "8000"]

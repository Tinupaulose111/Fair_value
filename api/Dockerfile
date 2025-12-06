# ----------------------------
# Stage 1: Builder
# ----------------------------
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

# Pip hygiene
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build dependencies (needed for scipy / peewee wheels if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY api/requirements-api.txt requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# ---------------------------------------------------------
# 🔒 CRITICAL FIX (DO NOT CHANGE):
# 1) Remove xgboost from requirements
# 2) Build wheels for everything else
# 3) Download CPU-only XGBoost wheel separately (NO DEPS)
# ---------------------------------------------------------

# Remove xgboost completely from dependency graph
RUN sed -i '/xgboost==/d' requirements.txt

# Build wheels for all NON-XGBOOST dependencies
RUN pip wheel --no-cache-dir \
    --prefer-binary \
    --wheel-dir=/wheels \
    -r requirements.txt

# Download CPU-only XGBoost wheel (NO CUDA, NO NVIDIA)
RUN pip download \
    --only-binary=:all: \
    --no-deps \
    --dest /wheels \
    xgboost==3.1.2


# ----------------------------
# Stage 2: Runtime
# ----------------------------
FROM python:3.12-slim-bookworm
WORKDIR /app

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# ---------------------------------------------------------
# ✅ FINAL FIX:
# Install normal deps FIRST, then XGBoost with --no-deps
# ---------------------------------------------------------
COPY --from=builder /wheels /wheels

RUN ls /wheels | grep -v xgboost | xargs -I {} pip install --no-cache-dir /wheels/{} && \
    pip install --no-cache-dir --no-deps /wheels/xgboost-3.1.2-*.whl && \
    rm -rf /wheels

# Runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Application code
# ----------------------------
COPY api/app.py /app/
COPY api/templates /app/templates
COPY api/static /app/static
COPY api/entrypoint.sh /app/

# Shared code & model
COPY core /app/core
COPY model /app/model

# Permissions
RUN dos2unix /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["./entrypoint.sh"]

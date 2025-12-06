# ----------------------------
# Stage 1: Builder
# ----------------------------
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# MUST use path within build context
COPY drift/requirements-locked.txt requirements.txt

RUN pip install --upgrade pip
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ----------------------------
# Stage 2: Runtime
# ----------------------------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Install Python wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy drift monitor script
COPY drift/drift_monitor.py /app/

# Copy shared modules and model from project root
COPY core /app/core
COPY model /app/model

# Fix permissions
RUN chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1

CMD ["python", "drift_monitor.py"]

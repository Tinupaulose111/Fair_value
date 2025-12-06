# -------------------- Stage 1: Builder --------------------
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy locked requirements (correct relative path)
COPY trainer/requirements-locked.txt requirements.txt

RUN pip install --upgrade pip
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# -------------------- Stage 2: Runtime --------------------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Install from wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy training code and dependencies
COPY trainer /app/trainer
COPY core /app/core
COPY model /app/model

# Give write permission (IMPORTANT: allows retraining to overwrite model)
RUN chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1

# Retraining script entry
CMD ["python", "trainer/retrain_model.py"]

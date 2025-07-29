FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy your app code
COPY . .
COPY firebase_credentials.json .

# ─────────────────────────────────────────────
# Runtime image (same base as builder)
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy installed packages and app code
COPY --from=builder /usr/local /usr/local 

COPY --from=builder /app /app 

# Create non-root user
RUN groupadd -r appgroup && useradd -m -s /bin/bash -u 1000 -g appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
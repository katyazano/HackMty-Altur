# ==============================================================================
# Altur Banking Voice Anti-Spoofing API - Production Dockerfile
# Python 3.12 Slim + Audio DSP (libsndfile, ffmpeg) + PyTorch CPU + XGBoost
# ==============================================================================

FROM python:3.12-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set application runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install essential system libraries for audio DSP, OpenMP, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime directories
RUN mkdir -p .cache Data weights

# Copy application source code, weights, and web assets
COPY src/ ./src/
COPY weights/ ./weights/
COPY web/ ./web/
COPY tests/ ./tests/
COPY app.py ./

# Expose HTTP port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server with Uvicorn
CMD ["python", "app.py"]

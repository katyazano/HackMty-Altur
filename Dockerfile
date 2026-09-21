FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system audio DSP dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (fast download ~180MB, zero CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install lean runtime requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dev requirements & evaluation notebooks (for on-demand dev installation)
COPY requirements-dev.txt ./
COPY model_evaluation_and_justification.ipynb ./

# Copy source code, pre-trained weights, static UI, scripts, and tests
COPY src/ ./src/
COPY weights/ ./weights/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY Data/manifest.csv ./Data/manifest.csv
COPY model.pkl threshold.json* ./
COPY app.py ./

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

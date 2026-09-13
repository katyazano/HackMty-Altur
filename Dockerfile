FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install CPU-only PyTorch to prevent downloading 3+ GB of unused NVIDIA CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install lean application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code and model artifacts
COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

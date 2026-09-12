# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Ensure standard output is not buffered
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface

# Install required system libraries for audio signal processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime storage directories
RUN mkdir -p demo_real_audios/processed_output sample_packs demo_data .cache

# Copy application source code, web assets, and bundled sample packs
COPY src/ ./src/
COPY web/ ./web/
COPY sample_packs/ ./sample_packs/
COPY app.py benchmark.py ./

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "app.py"]

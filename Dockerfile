# AI Scribe - Offline Medical Transcription Docker Image
# Supports both GPU (CUDA) and CPU deployment

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (for better caching)
RUN pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    libsox3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user (security best practice)
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/output /app/cache && \
    chown -R appuser:appuser /app

# Set up model cache directory
ENV HF_HOME="/app/cache"
ENV TRANSFORMERS_CACHE="/app/cache"

# Switch to app user
USER appuser

# Create volumes for persistent data
VOLUME ["/app/data", "/app/models", "/app/output"]

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "api_server.py"] 
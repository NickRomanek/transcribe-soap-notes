# 🐳 AI Scribe Docker Deployment Guide

**One-click deployment for AI Scribe with automatic GPU detection**

## 🚀 Quick Start (Recommended)

### For Linux/macOS:
```bash
# Make script executable and run
chmod +x docker-run.sh
./docker-run.sh
```

### For Windows:
```powershell
# Run in PowerShell
.\docker-run.ps1
```

## 📋 Prerequisites

### Required:
- **Docker** (20.10+) - [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (2.0+) - Usually included with Docker Desktop

### Optional (for GPU acceleration):
- **NVIDIA Docker Runtime** - [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **NVIDIA GPU** with 8GB+ VRAM (RTX 2070+)
- **CUDA-compatible drivers**

## 🔧 Installation Steps

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd ai-scribe
```

### 2. Configure Environment
```bash
# Copy environment template
cp env.sample .env

# Edit .env with your settings
# Required: HF_TOKEN (HuggingFace token)
# Optional: CUDA_VISIBLE_DEVICES, LOG_LEVEL
```

### 3. Get HuggingFace Token
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **read** permissions
3. Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. Add to `.env` file: `HF_TOKEN=your_token_here`

### 4. Run Deployment Script
```bash
./docker-run.sh    # Linux/macOS
.\docker-run.ps1   # Windows
```

## 📚 Deployment Options

### 1. API Server (Default)
**Best for:** Integration with other applications
```bash
docker-compose --profile api up -d
```
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:8000/app

### 2. Web Interface
**Best for:** User-friendly drag-and-drop interface
```bash
docker-compose --profile web up -d
```
- **Web Interface**: http://localhost:7860

### 3. CLI Interface
**Best for:** Batch processing and automation
```bash
docker-compose run --rm ai-scribe-cli bash
```
- Interactive shell with AI Scribe CLI tools

### 4. CPU-Only Mode
**Best for:** Systems without GPU
```bash
docker-compose --profile cpu up -d
```
- Automatically uses CPU-only inference

### 5. Development Mode
**Best for:** Development and testing
```bash
docker-compose --profile dev up -d
```
- **API Server**: http://localhost:8000
- **Jupyter Lab**: http://localhost:8888

## 🖥️ GPU Configuration

### Automatic Detection
The deployment scripts automatically detect GPU capabilities:
- ✅ **NVIDIA GPU + Docker Runtime** → GPU acceleration
- ⚠️ **No GPU/Driver Issues** → CPU fallback

### Manual GPU Configuration
```bash
# Force GPU usage
export CUDA_VISIBLE_DEVICES=0

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
```

### GPU Requirements
- **NVIDIA GPU** with 8GB+ VRAM
- **CUDA 11.8+ compatible drivers**
- **NVIDIA Container Runtime** installed

## 📁 Directory Structure

```
ai-scribe/
├── data/           # Input audio files
├── models/         # Downloaded AI models (persistent)
├── output/         # Processing results
├── docker-compose.yml
├── Dockerfile
├── docker-run.sh   # Linux/macOS launcher
├── docker-run.ps1  # Windows launcher
└── .env           # Your configuration
```

## 🔄 Common Usage Patterns

### Processing Audio Files
```bash
# 1. Place audio files in ./data/
cp your_audio.wav ./data/

# 2. Start API server
docker-compose --profile api up -d

# 3. Upload via web interface or API
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@./data/your_audio.wav"
```

### CLI Processing
```bash
# Interactive shell
docker-compose run --rm ai-scribe-cli bash

# Inside container:
python cli.py data/your_audio.wav
python cli.py list-results
```

### Batch Processing
```bash
# Process multiple files
for file in ./data/*.wav; do
  docker-compose run --rm ai-scribe-cli python cli.py "$file"
done
```

## 🛠️ Advanced Configuration

### Custom Docker Compose
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  ai-scribe-api:
    environment:
      - CUSTOM_SETTING=value
    ports:
      - "8080:8000"  # Custom port
```

### Resource Limits
```yaml
services:
  ai-scribe-api:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
```

### Custom Model Paths
```yaml
services:
  ai-scribe-api:
    volumes:
      - /path/to/your/models:/app/models
```

## 🚨 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Runtime
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 2. Permission Errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data ./models ./output
```

#### 3. Out of Memory
```bash
# Reduce batch size or use CPU mode
export CUDA_VISIBLE_DEVICES=""
docker-compose --profile cpu up -d
```

#### 4. Port Conflicts
```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

#### 5. Model Download Issues
```bash
# Pre-download models
docker-compose --profile download up ai-scribe-download

# Check HuggingFace token
docker-compose run --rm ai-scribe-cli python -c "import os; print(os.getenv('HF_TOKEN'))"
```

### Performance Optimization

#### GPU Memory
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Reduce model precision
# Edit ai_scribe/config.py:
# COMPUTE_TYPE = "int8"  # vs "float16"
```

#### CPU Usage
```bash
# Increase batch size for CPU
# Edit ai_scribe/config.py:
# BATCH_SIZE = 16  # vs 8
```

## 📈 Monitoring

### Container Logs
```bash
# View logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f ai-scribe-api
```

### Resource Usage
```bash
# Container stats
docker stats

# GPU usage
nvidia-smi
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 🔒 Security Considerations

### Production Deployment
```yaml
# docker-compose.prod.yml
services:
  ai-scribe-api:
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Network Security
```yaml
services:
  ai-scribe-api:
    networks:
      - internal
networks:
  internal:
    driver: bridge
```

### Data Protection
- Mount volumes with appropriate permissions
- Use secrets management for tokens
- Regular security updates

## 🎯 Use Cases

### Medical Practice
```bash
# HIPAA-compliant local deployment
docker-compose --profile cpu up -d  # CPU-only for security
```

### Research Institution
```bash
# High-performance GPU cluster
docker-compose --profile api up -d
```

### Development Team
```bash
# Full development environment
docker-compose --profile dev up -d
```

## 📞 Support

### Getting Help
1. Check the [troubleshooting section](#troubleshooting)
2. View container logs: `docker-compose logs`
3. Check GPU: `nvidia-smi`
4. Verify environment: `cat .env`

### Reporting Issues
Include this information:
```bash
# System info
docker version
docker-compose version
nvidia-smi  # If using GPU

# Container status
docker ps -a
docker-compose logs --tail=50
```

## 🔄 Updates

### Updating AI Scribe
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Updating Models
```bash
# Re-download models
docker-compose --profile download up ai-scribe-download --force-recreate
```

---

## 🎉 Quick Deploy Commands

```bash
# One-liner for API server
curl -sSL https://raw.githubusercontent.com/your-repo/ai-scribe/main/docker-run.sh | bash

# Or clone and run
git clone <repo-url> && cd ai-scribe && ./docker-run.sh
```

**That's it! Your AI Scribe instance should be running and ready to process audio files.** 🚀 
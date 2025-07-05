# 🚀 AI Scribe Quick Setup Guide

## ✅ Current Status
Your AI Scribe project is **ready to use** with basic transcription functionality!

### What's Working:
- ✅ Virtual environment created
- ✅ CLI interface functional
- ✅ Core transcription (faster-whisper) installed
- ✅ Basic dependencies installed

## 🔧 Daily Usage

### 1. Activate Virtual Environment
```bash
.\venv\Scripts\Activate.ps1
```

### 2. Test CLI Commands
```bash
# Show help
python cli.py --help

# Validate audio file
python cli.py validate data/your_audio.wav

# List recent results
python cli.py list-results
```

### 3. Process Audio (Basic)
```bash
# Add audio file to data/ directory first
python cli.py data/your_audio.wav
```

## 🔄 Optional: Full Functionality

### Install Additional Dependencies
```bash
# Speaker diarization (requires HF token)
pip install pyannote.audio==3.1.*

# SOAP note generation
pip install llama-cpp-python

# Web interface
pip install gradio
```

### Set up HuggingFace Token
1. Get token from: https://huggingface.co/settings/tokens
2. Accept terms for: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create `.env` file:
```bash
copy .env.sample .env
# Edit .env and add: HF_TOKEN=your_token_here
```

### Download Phi-3 Model (for SOAP notes)
```bash
# Download to models/ directory
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
```

## 📁 Project Structure
```
ai_scribe/
├── venv/                    # Virtual environment
├── data/                    # Put your audio files here
├── models/                  # Downloaded models
├── output/                  # Processing results
├── ai_scribe/              # Core package
├── cli.py                  # Command line interface
└── README.md               # Full documentation
```

## 🎯 Quick Start Commands
```bash
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Add audio file to data/
# 3. Process it
python cli.py data/your_audio.wav

# 4. Check results
python cli.py list-results
```

## 🆘 Troubleshooting

### Poetry Issues
- Use `pip` instead of `poetry` (already set up)
- Virtual environment is working fine

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Audio Format Issues
- Supported: WAV, MP3, M4A, FLAC
- Convert with: `ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav`

### Memory Issues
- Reduce batch size in `ai_scribe/config.py`
- Use CPU: set `CUDA_VISIBLE_DEVICES=""` in `.env`

## 📞 Next Steps
1. Add a test audio file to `data/`
2. Run: `python cli.py data/your_audio.wav`
3. Check results in `output/` directory
4. Explore the notebook: `jupyter lab notebooks/`

**🎉 You're all set! The basic transcription pipeline is ready to use.** 
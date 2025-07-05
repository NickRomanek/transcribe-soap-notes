"""
Configuration settings for AI Scribe
"""

from pathlib import Path
import os

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Base paths
BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
OUTPUT_DIR = BASE / "output"

# Model paths
WHISPER_MODEL = MODELS_DIR / "whisper-large-v3-int8"
DIAR_MODEL = MODELS_DIR / "pyannote_speaker"
LLM_MODEL = MODELS_DIR / "Phi-3-mini-4k-instruct-q4.gguf"

# Separate model for speaker role analysis (optional)
ROLE_ANALYSIS_MODEL = MODELS_DIR / "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"  # Smaller, faster for role detection
USE_SEPARATE_ROLE_MODEL = False  # Set to True to use separate model for role analysis

# Model configuration
SUMMARY_MODEL_CONFIG = {
    "n_ctx": 4096,
    "n_batch": 512,
    "temperature": 0.1
}

ROLE_MODEL_CONFIG = {
    "n_ctx": 2048,  # Smaller context for role analysis
    "n_batch": 256,
    "temperature": 0.05  # More deterministic for role detection
}

# Processing settings
CHUNK_SEC = 30
SAMPLE_RATE = 16000
BATCH_SIZE = 8

# Output retention (days)
RETENTION_DAYS = 30

# Hugging Face token (for pyannote models)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# CUDA settings
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True) 
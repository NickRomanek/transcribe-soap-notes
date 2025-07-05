#!/usr/bin/env python3
"""
Script to download and cache models for AI_Scribe
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def download_file_with_progress(url, destination, expected_hash=None):
    """Download a file with progress bar and hash verification"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = hashlib.sha256(open(destination, 'rb').read()).hexdigest()
            if actual_hash != expected_hash:
                print(f"⚠️  Hash mismatch for {destination.name}")
                print(f"   Expected: {expected_hash}")
                print(f"   Actual:   {actual_hash}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False

def download_whisper_model():
    """Download and cache the Whisper model"""
    print("📥 Downloading Whisper model...")
    try:
        from faster_whisper import WhisperModel
        
        # This will download the model to the cache directory
        model = WhisperModel("large-v3", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        print("✓ Whisper model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading Whisper model: {e}")
        return False

def download_pyannote_model():
    """Download and cache the pyannote model"""
    print("📥 Downloading pyannote model...")
    try:
        from pyannote.audio import Pipeline
        
        # This will download the model to the cache directory
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        print("✓ Pyannote model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading pyannote model: {e}")
        if "access token" in str(e).lower():
            print("   💡 Make sure to set your HF_TOKEN environment variable")
            print("   💡 Get a token from: https://huggingface.co/settings/tokens")
        return False

def download_phi3_model():
    """Download Phi-3 Mini model"""
    print("📥 Downloading Phi-3 Mini model...")
    
    model_path = Path("models/Phi-3-mini-4k-instruct-q4.gguf")
    
    if model_path.exists():
        print("✓ Phi-3 model already exists")
        return True
    
    url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    
    success = download_file_with_progress(url, model_path)
    if success:
        print("✓ Phi-3 model downloaded successfully")
    return success

def download_yolo_models():
    """Download YOLO models"""
    print("📥 Downloading YOLO models...")
    
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
    }
    
    success_count = 0
    for model_name, url in models.items():
        model_path = Path(f"models/{model_name}")
        
        if model_path.exists():
            print(f"✓ {model_name} already exists")
            success_count += 1
            continue
        
        if download_file_with_progress(url, model_path):
            print(f"✓ {model_name} downloaded successfully")
            success_count += 1
    
    return success_count == len(models)

def check_disk_space():
    """Check if there's enough disk space for models"""
    required_space_gb = 10  # Rough estimate
    
    try:
        import shutil
        free_space = shutil.disk_usage(Path.cwd()).free / (1024**3)
        
        if free_space < required_space_gb:
            print(f"⚠️  Warning: Low disk space ({free_space:.1f}GB free, {required_space_gb}GB recommended)")
            return False
        
        print(f"✓ Sufficient disk space ({free_space:.1f}GB free)")
        return True
    except:
        print("⚠️  Could not check disk space")
        return True

def main():
    print("🚀 AI Scribe Model Downloader")
    print("=" * 50)
    
    # Check prerequisites
    if not check_disk_space():
        print("❌ Insufficient disk space. Please free up space and try again.")
        return
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Download models
    print("\n📦 Downloading models...")
    results = {
        "Whisper": download_whisper_model(),
        "Pyannote": download_pyannote_model(),
        "Phi-3": download_phi3_model(),
        "YOLO": download_yolo_models()
    }
    
    print("\n" + "=" * 50)
    print("📊 Download Results:")
    
    success_count = 0
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")
        if success:
            success_count += 1
    
    print(f"\n🎯 {success_count}/{len(results)} models downloaded successfully")
    
    if success_count == len(results):
        print("\n🎉 All models ready! You can now run the full pipeline.")
        print("💡 Try: python cli.py data/example.wav")
    else:
        print("\n❌ Some models failed to download.")
        print("💡 Check your internet connection and try again.")
        print("💡 For pyannote models, make sure HF_TOKEN is set in your environment.")

if __name__ == "__main__":
    main() 
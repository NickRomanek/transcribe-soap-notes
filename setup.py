#!/usr/bin/env python3
"""
Setup script for AI Scribe - handles initial setup and model downloads
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed: {result.stderr}")
            return False
        print(f"✅ Success!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'output', 'logs']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created {dir_name}/ directory")

def setup_environment():
    """Setup environment file"""
    env_file = Path('.env')
    env_sample = Path('env.sample')
    
    if not env_file.exists() and env_sample.exists():
        import shutil
        shutil.copy(env_sample, env_file)
        print("📝 Created .env file from template")
        print("💡 Please edit .env and add your HuggingFace token")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("# HuggingFace token for pyannote models\n")
            f.write("HF_TOKEN=your_token_here\n")
            f.write("\n# Optional: Force CPU usage\n")
            f.write("# CUDA_VISIBLE_DEVICES=\n")
        print("📝 Created basic .env file")
        print("💡 Please edit .env and add your HuggingFace token")

def main():
    print("🚀 AI Scribe Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_environment()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return
    
    # Download models
    print("\n🔽 Downloading models...")
    if not run_command(f"{sys.executable} download_models.py", "Downloading AI models"):
        print("⚠️  Model download failed. You can run 'python download_models.py' manually later.")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("=" * 50)
    print("📖 Next steps:")
    print("1. Edit .env file and add your HuggingFace token")
    print("2. Test with: python cli.py --help")
    print("3. Process audio: python cli.py data/your_audio.wav")
    print("4. Launch web UI: python app_gradio.py")
    print("=" * 50)

if __name__ == "__main__":
    main() 
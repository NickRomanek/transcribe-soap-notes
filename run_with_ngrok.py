#!/usr/bin/env python3
"""
Script to run AI Scribe project with ngrok tunnel
"""

import os
import sys
import time
import subprocess
import signal
import threading
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NgrokManager:
    def __init__(self, port: int = 8000):
        self.port = port
        self.ngrok_process = None
        self.api_url = "http://localhost:4040/api"
        self.tunnel_url = None
        
    def start_ngrok(self) -> bool:
        """Start ngrok tunnel"""
        try:
            # Check if ngrok is installed
            result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("ngrok is not installed or not in PATH")
                logger.info("Please install ngrok from https://ngrok.com/download")
                return False
            
            # Start ngrok tunnel
            logger.info(f"Starting ngrok tunnel for port {self.port}")
            self.ngrok_process = subprocess.Popen(
                ['ngrok', 'http', str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for ngrok to start
            time.sleep(3)
            
            # Get tunnel URL
            self.tunnel_url = self.get_tunnel_url()
            if self.tunnel_url:
                logger.info(f"✅ ngrok tunnel started: {self.tunnel_url}")
                return True
            else:
                logger.error("Failed to get ngrok tunnel URL")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            return False
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get the public ngrok URL"""
        try:
            response = requests.get(f"{self.api_url}/tunnels", timeout=5)
            if response.status_code == 200:
                tunnels = response.json()['tunnels']
                for tunnel in tunnels:
                    if tunnel['proto'] == 'https':
                        return tunnel['public_url']
            return None
        except Exception as e:
            logger.error(f"Failed to get tunnel URL: {e}")
            return None
    
    def stop_ngrok(self):
        """Stop ngrok tunnel"""
        if self.ngrok_process:
            logger.info("Stopping ngrok tunnel...")
            self.ngrok_process.terminate()
            try:
                self.ngrok_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ngrok_process.kill()
            logger.info("ngrok tunnel stopped")

class FastAPIServer:
    def __init__(self, port: int = 8000):
        self.port = port
        self.server_process = None
        
    def start_server(self) -> bool:
        """Start FastAPI server"""
        try:
            logger.info(f"Starting FastAPI server on port {self.port}")
            
            # Check if required dependencies are installed
            try:
                import fastapi
                import uvicorn
            except ImportError:
                logger.error("FastAPI or uvicorn not installed")
                logger.info("Installing required dependencies...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'fastapi', 'uvicorn'], check=True)
            
            # Start the server
            self.server_process = subprocess.Popen(
                [sys.executable, 'api_server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            if self.is_server_running():
                logger.info(f"✅ FastAPI server started on port {self.port}")
                return True
            else:
                logger.error("Failed to start FastAPI server")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_server(self):
        """Stop FastAPI server"""
        if self.server_process:
            logger.info("Stopping FastAPI server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            logger.info("FastAPI server stopped")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, check=True)
        logger.info("Dependencies installed successfully")

def main():
    """Main function to run the project with ngrok"""
    print("🚀 Starting AI Scribe with ngrok...")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Initialize components
    server = FastAPIServer(port=8000)
    ngrok = NgrokManager(port=8000)
    
    # Handle cleanup on exit
    def cleanup(signum=None, frame=None):
        logger.info("\n🛑 Shutting down...")
        ngrok.stop_ngrok()
        server.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Start FastAPI server
        if not server.start_server():
            logger.error("Failed to start FastAPI server")
            return 1
        
        # Start ngrok tunnel
        if not ngrok.start_ngrok():
            logger.error("Failed to start ngrok tunnel")
            server.stop_server()
            return 1
        
        # Display information
        print("\n" + "=" * 50)
        print("🎉 AI Scribe is now running!")
        print("=" * 50)
        print(f"📡 Local API: http://localhost:8000")
        print(f"🌐 Public API: {ngrok.tunnel_url}")
        print(f"📱 Frontend: {ngrok.tunnel_url}/app")
        print(f"🔍 API Docs: {ngrok.tunnel_url}/docs")
        print(f"📊 Health Check: {ngrok.tunnel_url}/health")
        print("=" * 50)
        print("Press Ctrl+C to stop the servers")
        print("=" * 50)
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
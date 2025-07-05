#!/usr/bin/env python3
"""
FastAPI server for AI Scribe
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional
import shutil
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import ai_scribe.pipeline as pipeline
from ai_scribe.config import OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Scribe API",
    description="Offline AI transcription, diarization, and summarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_path = Path("frontend")
if frontend_path.exists() and frontend_path.is_dir():
    logger.info(f"Mounting static files from: {frontend_path.absolute()}")
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
else:
    logger.warning(f"Frontend directory not found: {frontend_path.absolute()}")

class ProcessingStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None
    upload_progress: Optional[float] = None

# Store processing status
processing_status = {}

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AI Scribe API is running"}

@app.get("/app")
async def serve_frontend():
    """Serve the main frontend application"""
    frontend_file = frontend_path / "index.html"
    logger.info(f"Attempting to serve frontend from: {frontend_file.absolute()}")
    if frontend_file.exists() and frontend_file.is_file():
        return FileResponse(str(frontend_file))
    else:
        logger.error(f"Frontend file not found: {frontend_file.absolute()}")
        raise HTTPException(status_code=404, detail=f"Frontend not found at {frontend_file}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": True}  # You can add actual GPU check

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_dir: Optional[str] = None
):
    """
    Transcribe an audio file
    """
    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Generate job ID
    job_id = f"job_{int(time.time())}_{len(processing_status) + 1}"
    
    # Read the entire file into memory to avoid timeouts
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    # Set initial status
    processing_status[job_id] = {
        "status": "processing", 
        "message": "File uploaded. Queued for processing...",
        "upload_progress": 100.0,
        "progress": 0.0
    }
    
    # Start background processing with the file bytes
    background_tasks.add_task(
        process_audio_bytes,
        job_id,
        file_bytes,
        file.filename,
        output_dir
    )
    
    return {
        "job_id": job_id,
        "message": "File uploaded successfully, processing has started.",
        "status": "processing"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get processing status
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Get processing results
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = processing_status[job_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not complete")
    
    # Return results
    return status.get("results", {})

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a file from the output directory
    """
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=str(file_path), 
        media_type='application/octet-stream', 
        filename=filename
    )

@app.get("/output-files")
async def list_output_files():
    """
    List all files in the output directory with their metadata
    """
    try:
        files = []
        for file_path in OUTPUT_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "extension": file_path.suffix.lower()
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")

# New background task function
async def process_audio_bytes(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    output_dir: Optional[str] = None
):
    """
    Saves file from bytes and runs the processing pipeline
    """
    data_file_path = DATA_DIR / filename
    try:
        # Save file from memory to disk
        with open(data_file_path, "wb") as f:
            f.write(file_bytes)

        processing_status[job_id]["message"] = "Loading models..."
        processing_status[job_id]["progress"] = 5.0

        # Validate audio file
        pipeline.validate_audio_file(data_file_path)
        
        output_path = Path(output_dir) if output_dir else OUTPUT_DIR
        
        # Define progress callback function
        def update_progress(progress: float, message: str):
            processing_status[job_id]["progress"] = progress
            processing_status[job_id]["message"] = message
            logger.info(f"Job {job_id}: {progress:.1f}% - {message}")
        
        # Run pipeline with progress callback
        results = pipeline.run(str(data_file_path), output_path, progress_callback=update_progress)
        
        # Update final status
        processing_status[job_id] = {
            "status": "completed",
            "message": "✅ Processing complete!",
            "progress": 100.0,
            "upload_progress": 100.0,
            "results": results,
            "file_path": str(data_file_path),
            "output_files": {
                "json_file": f"{Path(filename).stem}.json",
                "txt_file": f"{Path(filename).stem}.txt"
            }
        }
        logger.info(f"Job {job_id} completed successfully. Files saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        processing_status[job_id] = {
            "status": "error",
            "message": f"❌ Processing failed: {str(e)}",
            "progress": 0.0,
            "upload_progress": 0.0
        }
        # Clean up file if it exists
        if data_file_path.exists():
            data_file_path.unlink()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
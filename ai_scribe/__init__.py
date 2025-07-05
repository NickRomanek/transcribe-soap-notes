"""
AI Scribe - Offline transcription, diarization, and summarization
"""

__version__ = "0.1.0"

from . import config
from . import transcribe
from . import diarize
from . import align
from . import summarise
from . import pipeline

__all__ = [
    "config",
    "transcribe", 
    "diarize",
    "align",
    "summarise",
    "pipeline",
] 
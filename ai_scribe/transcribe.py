"""
Transcription module using faster-whisper
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging
import os

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from . import config

logger = logging.getLogger(__name__)


class Transcriber:
    """Whisper-based transcription using faster-whisper"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.WHISPER_MODEL
        self.model = None
        
    def load_model(self):
        """Load the Whisper model"""
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed")
            
        logger.info(f"Loading Whisper model...")
        
        # Force offline mode to use cached models only
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        try:
            # Try to load from cache first (offline mode)
            self.model = WhisperModel(
                "large-v3",
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                local_files_only=True  # This forces offline mode
            )
            logger.info("Whisper model loaded from cache (offline)")
        except Exception as e:
            # If cache fails, fall back to local path or download
            logger.warning(f"Failed to load from cache: {e}")
            os.environ.pop("HF_HUB_OFFLINE", None)  # Remove offline constraint
            
            if self.model_path.exists():
                logger.info(f"Loading from local path: {self.model_path}")
                model_name = str(self.model_path)
            else:
                logger.info("Loading from HuggingFace (will download if needed)")
                model_name = "large-v3"
                
            self.model = WhisperModel(
                model_name,
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                download_root=str(config.MODELS_DIR)
            )
            logger.info("Whisper model loaded successfully")
        
    def transcribe(self, wav_path: Path) -> List[Dict]:
        """
        Transcribe audio file to segments with timestamps
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            List of dictionaries with 'start', 'end', 'text' keys
        """
        if self.model is None:
            self.load_model()
            
        logger.info(f"Transcribing {wav_path}")
        
        segments, info = self.model.transcribe(
            str(wav_path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        results = []
        for segment in segments:
            # Extract word-level timestamps if available
            words = []
            if hasattr(segment, 'words') and segment.words:
                words = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in segment.words
                ]
            
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob
            })
            
        logger.info(f"Transcription complete: {len(results)} segments")
        return results


def transcribe(wav_path: Path) -> List[Dict]:
    """
    Convenience function for transcription
    
    Args:
        wav_path: Path to WAV audio file
        
    Returns:
        List of transcription segments
    """
    transcriber = Transcriber()
    return transcriber.transcribe(wav_path) 
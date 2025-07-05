"""
Speaker diarization module using pyannote.audio v3.1
"""

from typing import List, Tuple, Optional
from pathlib import Path
import logging
import os

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError:
    Pipeline = None
    Annotation = None
    Segment = None

from . import config

logger = logging.getLogger(__name__)


class Diarizer:
    """Speaker diarization using pyannote.audio"""
    
    def __init__(self, model_path: Optional[Path] = None, model_version: str = "3.1"):
        self.model_path = model_path or config.DIAR_MODEL
        self.model_version = model_version
        self.pipeline = None
        
    def load_model(self):
        """Load the pyannote diarization pipeline"""
        if Pipeline is None:
            raise ImportError("pyannote.audio is not installed")
            
        # Try different model versions
        model_names = [
            f"pyannote/speaker-diarization-{self.model_version}",
            "pyannote/speaker-diarization-3.1",   # Latest version first
            "pyannote/speaker-diarization-3.0"    # Fallback to older version
        ]
        
        logger.info(f"Loading pyannote speaker diarization pipeline (trying version {self.model_version})")
        
        # Force offline mode to use cached models only
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        for model_name in model_names:
            try:
                # Try to load from cache first (offline mode) - without local_files_only for compatibility
                self.pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=config.HF_TOKEN if config.HF_TOKEN else None,
                    cache_dir=str(config.MODELS_DIR)
                )
                logger.info(f"Diarization pipeline loaded from cache (offline): {model_name}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name} from cache: {e}")
                continue
        
        if self.pipeline is None:
            # If cache fails, try with online mode
            logger.warning("Failed to load from cache, trying online mode")
            os.environ.pop("HF_HUB_OFFLINE", None)  # Remove offline constraint
            
            for model_name in model_names:
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=config.HF_TOKEN if config.HF_TOKEN else None,
                        cache_dir=str(config.MODELS_DIR)
                    )
                    logger.info(f"Diarization pipeline loaded from HuggingFace: {model_name}")
                    break
                    
                except Exception as e2:
                    logger.warning(f"Failed to load {model_name} from HuggingFace: {e2}")
                    continue
        
        if self.pipeline is None:
            logger.info("Falling back to local model if available")
            if self.model_path.exists():
                self.pipeline = Pipeline.from_pretrained(str(self.model_path))
            else:
                raise RuntimeError("No diarization model available. Please set HF_TOKEN or download model locally.")
        
        # Move to GPU if available
        if config.DEVICE == "cuda" and self.pipeline is not None:
            import torch
            self.pipeline = self.pipeline.to(torch.device("cuda"))
                
        logger.info("Diarization pipeline loaded successfully")
    
    def _consolidate_short_segments(self, segments: List[Tuple[float, float, str]], min_duration: float = 1.0) -> List[Tuple[float, float, str]]:
        """
        Consolidate very short segments with adjacent segments from the same speaker
        This reduces over-segmentation while preserving speaker changes
        """
        if not segments:
            return segments
            
        consolidated = []
        current_segment = list(segments[0])  # [start, end, speaker]
        
        for next_segment in segments[1:]:
            next_start, next_end, next_speaker = next_segment
            curr_start, curr_end, curr_speaker = current_segment
            
            # Check if we should merge
            segment_duration = curr_end - curr_start
            next_duration = next_end - next_start
            gap = next_start - curr_end
            
            should_merge = (
                # Same speaker
                next_speaker == curr_speaker and
                # Small gap between segments (up to 2 seconds)
                gap < 2.0 and
                # At least one segment is short OR both are moderate length
                (segment_duration < min_duration or next_duration < min_duration or 
                 (segment_duration < 3.0 and next_duration < 3.0))
            )
            
            if should_merge:
                # Extend current segment
                current_segment[1] = next_end  # Update end time
            else:
                # Save current segment and start new one
                consolidated.append(tuple(current_segment))
                current_segment = list(next_segment)
        
        # Add the last segment
        consolidated.append(tuple(current_segment))
        
        return consolidated
    
    def _improve_speaker_boundaries_conservative(self, segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """
        More conservative approach to improving speaker boundaries
        Only split segments that are very long and likely to contain multiple speakers
        """
        improved_segments = []
        
        for start, end, speaker in segments:
            duration = end - start
            
            # Only split very long segments (>6 seconds) and only if we have multiple speakers
            all_speakers = set(s[2] for s in segments)
            if duration > 6.0 and len(all_speakers) > 1:
                # Split into 2-3 chunks max
                num_chunks = min(3, int(duration / 3.0))
                chunk_duration = duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start + (i * chunk_duration)
                    chunk_end = min(start + ((i + 1) * chunk_duration), end)
                    
                    # Alternate speakers more conservatively
                    if i % 2 == 0:
                        chunk_speaker = speaker
                    else:
                        # Use a different speaker if available
                        other_speakers = [s for s in all_speakers if s != speaker]
                        chunk_speaker = other_speakers[0] if other_speakers else speaker
                    
                    improved_segments.append((chunk_start, chunk_end, chunk_speaker))
            else:
                improved_segments.append((start, end, speaker))
        
        return improved_segments
        
    def _fix_speaker_consistency(self, segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """
        Fix speaker consistency by looking at longer patterns and reducing rapid switches
        """
        if len(segments) < 3:
            return segments
            
        improved = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            start, end, speaker = current
            
            # Look ahead for rapid speaker switches (within 3 seconds)
            rapid_switch_window = []
            j = i
            window_end_time = start + 4.0  # 4-second window
            
            while j < len(segments) and segments[j][0] < window_end_time:
                rapid_switch_window.append(j)
                j += 1
            
            # If we have rapid switches in this window, try to smooth them
            if len(rapid_switch_window) > 3:  # More than 3 segments in 4 seconds
                # Find the dominant speaker in this window
                speaker_durations = {}
                for idx in rapid_switch_window:
                    seg_start, seg_end, seg_speaker = segments[idx]
                    duration = seg_end - seg_start
                    speaker_durations[seg_speaker] = speaker_durations.get(seg_speaker, 0) + duration
                
                # Get the speaker with the most time
                dominant_speaker = max(speaker_durations.keys(), key=lambda x: speaker_durations[x])
                
                # Merge all segments in this window to the dominant speaker
                window_start = segments[rapid_switch_window[0]][0]
                window_end = segments[rapid_switch_window[-1]][1]
                improved.append((window_start, window_end, dominant_speaker))
                
                # Skip all segments in this window
                i = rapid_switch_window[-1] + 1
            else:
                # No rapid switching, keep the segment as is
                improved.append(current)
                i += 1
        
        return improved
        
    def _smart_speaker_assignment(self, segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """
        Improve speaker assignment by looking at patterns and reducing random switches
        """
        if len(segments) < 3:
            return segments
            
        improved = []
        
        for i, (start, end, speaker) in enumerate(segments):
            # Look at neighboring segments
            prev_speaker = segments[i-1][2] if i > 0 else None
            next_speaker = segments[i+1][2] if i < len(segments)-1 else None
            
            # If this is a very short segment surrounded by the same speaker, reassign it
            duration = end - start
            if (duration < 2.0 and prev_speaker and next_speaker and 
                prev_speaker == next_speaker and speaker != prev_speaker):
                # Reassign to the surrounding speaker
                improved.append((start, end, prev_speaker))
            elif (duration < 0.5 and prev_speaker and speaker != prev_speaker):
                # Very short segments likely belong to previous speaker
                improved.append((start, end, prev_speaker))
            else:
                improved.append((start, end, speaker))
        
        return improved
        
    def diarize(self, wav_path: Path) -> List[Tuple[float, float, str]]:
        """
        Perform speaker diarization on audio file
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            List of tuples (start_time, end_time, speaker_label)
        """
        if self.pipeline is None:
            self.load_model()
            
        logger.info(f"Diarizing {wav_path}")
        
        # Run diarization with more conservative parameters
        try:
            # Use more conservative parameters to reduce over-segmentation
            diarization = self.pipeline(
                str(wav_path), 
                min_speakers=2, 
                max_speakers=3,
                # More conservative segmentation threshold (if supported)
                segmentation_threshold=0.8,  # Higher threshold = fewer segments
                clustering_threshold=0.7     # Higher threshold = fewer speaker changes
            )
        except TypeError:
            # Fallback with basic parameters if advanced params not supported
            try:
                diarization = self.pipeline(str(wav_path), min_speakers=2, max_speakers=3)
            except TypeError:
                # Final fallback
                diarization = self.pipeline(str(wav_path))
        
        # Convert to list of tuples
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append((
                turn.start,
                turn.end,
                speaker
            ))
            
        # Sort by start time
        results.sort(key=lambda x: x[0])
        
        # Post-process more conservatively
        logger.info(f"Raw diarization: {len(results)} segments")
        
        # Conservative boundary improvement (only for very long segments)
        results = self._improve_speaker_boundaries_conservative(results)
        logger.info(f"After conservative boundary improvement: {len(results)} segments")
        
        # Consolidate short segments
        results = self._consolidate_short_segments(results, min_duration=1.5)
        logger.info(f"After consolidating short segments: {len(results)} segments")
        
        # Fix speaker consistency
        results = self._fix_speaker_consistency(results)
        logger.info(f"After fixing speaker consistency: {len(results)} segments")
        
        # Smart speaker assignment
        results = self._smart_speaker_assignment(results)
        logger.info(f"After smart speaker assignment: {len(results)} segments")
        
        logger.info(f"Diarization complete: {len(results)} speaker segments")
        return results
        
    def get_speakers_count(self, wav_path: Path) -> int:
        """Get the number of unique speakers detected"""
        segments = self.diarize(wav_path)
        speakers = set(segment[2] for segment in segments)
        return len(speakers)


def diarize(wav_path: Path, model_version: str = "3.1") -> List[Tuple[float, float, str]]:
    """
    Convenience function for speaker diarization
    
    Args:
        wav_path: Path to WAV audio file
        model_version: Version of pyannote model to use ("3.0" or "3.1")
        
    Returns:
        List of speaker segments (start, end, speaker)
    """
    diarizer = Diarizer(model_version=model_version)
    return diarizer.diarize(wav_path) 
"""
Alignment module to merge transcription segments with speaker diarization
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def align_segments_advanced(
    transcription: List[Dict], 
    diarization: List[Tuple[float, float, str]]
) -> List[Dict]:
    """
    Advanced alignment that can split transcription segments when speaker changes occur
    
    Args:
        transcription: List of transcription segments with start, end, text
        diarization: List of speaker segments (start, end, speaker)
        
    Returns:
        List of aligned segments with speaker, text, start, end
    """
    logger.info(f"Aligning {len(transcription)} transcription segments with {len(diarization)} speaker segments")
    
    aligned_segments = []
    
    for trans_seg in transcription:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_text = trans_seg["text"]
        words = trans_seg.get("words", [])
        
        # Find all speaker segments that overlap with this transcription segment
        overlapping_speakers = []
        for diar_start, diar_end, speaker in diarization:
            # Check if there's any overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            
            if overlap_end > overlap_start:
                overlapping_speakers.append({
                    'start': diar_start,
                    'end': diar_end,
                    'speaker': speaker,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end
                })
        
        if not overlapping_speakers:
            # No speaker found, use unknown
            aligned_segments.append({
                "speaker": "UNKNOWN",
                "text": trans_text,
                "start": trans_start,
                "end": trans_end,
                "duration": trans_end - trans_start,
                "overlap_confidence": 0.0,
                "words": words
            })
            continue
        
        # Sort by overlap start time
        overlapping_speakers.sort(key=lambda x: x['overlap_start'])
        
        # If only one speaker or no speaker changes, use simple alignment
        if len(overlapping_speakers) == 1:
            speaker_info = overlapping_speakers[0]
            aligned_segments.append({
                "speaker": speaker_info['speaker'],
                "text": trans_text,
                "start": trans_start,
                "end": trans_end,
                "duration": trans_end - trans_start,
                "overlap_confidence": 1.0,
                "words": words
            })
        else:
            # Multiple speakers - try to split the text
            split_segments = _split_transcription_by_speakers(
                trans_seg, overlapping_speakers
            )
            aligned_segments.extend(split_segments)
    
    logger.info(f"Alignment complete: {len(aligned_segments)} aligned segments")
    return aligned_segments


def _split_transcription_by_speakers(
    trans_seg: Dict, 
    overlapping_speakers: List[Dict]
) -> List[Dict]:
    """
    Split a transcription segment based on speaker boundaries
    """
    trans_start = trans_seg["start"]
    trans_end = trans_seg["end"]
    trans_text = trans_seg["text"]
    words = trans_seg.get("words", [])
    
    segments = []
    
    # If we have word-level timestamps, use them for precise splitting
    if words:
        segments = _split_by_word_timestamps(trans_seg, overlapping_speakers)
    else:
        # Fallback: split text proportionally by time
        segments = _split_by_time_proportion(trans_seg, overlapping_speakers)
    
    return segments


def _split_by_word_timestamps(
    trans_seg: Dict, 
    overlapping_speakers: List[Dict]
) -> List[Dict]:
    """
    Split transcription using word-level timestamps with conservative sentence preservation
    """
    words = trans_seg["words"]
    segments = []
    
    # If the transcription segment is short (< 3 seconds), assign to dominant speaker
    trans_duration = trans_seg["end"] - trans_seg["start"]
    if trans_duration < 3.0:
        # Find the speaker with the most overlap
        best_speaker = "UNKNOWN"
        max_overlap = 0
        for speaker_info in overlapping_speakers:
            overlap_duration = speaker_info['overlap_end'] - speaker_info['overlap_start']
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker_info['speaker']
        
        return [{
            "speaker": best_speaker,
            "text": trans_seg["text"],
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "duration": trans_duration,
            "overlap_confidence": 1.0,
            "words": words
        }]
    
    current_speaker = None
    current_words = []
    current_start = None
    
    for i, word in enumerate(words):
        word_start = word.get("start", 0)
        word_end = word.get("end", 0)
        word_mid = (word_start + word_end) / 2
        word_text = word.get("word", "")
        
        # Find which speaker this word belongs to
        best_speaker = "UNKNOWN"
        for speaker_info in overlapping_speakers:
            if speaker_info['overlap_start'] <= word_mid <= speaker_info['overlap_end']:
                best_speaker = speaker_info['speaker']
                break
        
        # Check if this is a sentence boundary (look for punctuation)
        is_sentence_end = any(punct in word_text for punct in ['.', '!', '?', ','])
        next_word_text = words[i + 1].get("word", "") if i + 1 < len(words) else ""
        is_natural_break = (
            is_sentence_end or 
            word_text.endswith(',') or
            next_word_text.strip().startswith('"') or
            (i + 1 < len(words) and words[i + 1].get("start", 0) - word_end > 0.5)  # Long pause
        )
        
        # If speaker changed, only split at natural breaks
        if current_speaker != best_speaker and current_words:
            # Only split if we're at a natural break OR if we've been with wrong speaker for >2 seconds
            current_duration = word_start - current_start if current_start else 0
            
            if is_natural_break or current_duration > 2.0:
                # Create segment for previous speaker
                segment_text = "".join([w.get("word", "") for w in current_words])
                segments.append({
                    "speaker": current_speaker,
                    "text": segment_text.strip(),
                    "start": current_start,
                    "end": current_words[-1].get("end", current_start),
                    "duration": current_words[-1].get("end", current_start) - current_start,
                    "overlap_confidence": 1.0,
                    "words": current_words.copy()
                })
                current_words = []
            else:
                # Don't split - keep with current speaker
                best_speaker = current_speaker
        
        # Start new segment or continue current one
        if not current_words:
            current_start = word_start
            current_speaker = best_speaker
        
        current_words.append(word)
    
    # Add the last segment
    if current_words:
        segment_text = "".join([w.get("word", "") for w in current_words])
        segments.append({
            "speaker": current_speaker,
            "text": segment_text.strip(),
            "start": current_start,
            "end": current_words[-1].get("end", current_start),
            "duration": current_words[-1].get("end", current_start) - current_start,
            "overlap_confidence": 1.0,
            "words": current_words
        })
    
    return segments


def _split_by_time_proportion(
    trans_seg: Dict, 
    overlapping_speakers: List[Dict]
) -> List[Dict]:
    """
    Split transcription proportionally by time when no word timestamps available
    """
    trans_start = trans_seg["start"]
    trans_end = trans_seg["end"]
    trans_text = trans_seg["text"]
    trans_duration = trans_end - trans_start
    
    segments = []
    text_words = trans_text.split()
    
    for i, speaker_info in enumerate(overlapping_speakers):
        # Calculate what portion of the transcription this speaker covers
        speaker_start = max(speaker_info['overlap_start'], trans_start)
        speaker_end = min(speaker_info['overlap_end'], trans_end)
        speaker_duration = speaker_end - speaker_start
        
        if speaker_duration <= 0:
            continue
            
        # Calculate text portion
        portion = speaker_duration / trans_duration
        word_count = max(1, int(len(text_words) * portion))
        
        # Get words for this speaker
        start_word_idx = sum(
            max(1, int(len(text_words) * 
                ((overlapping_speakers[j]['overlap_end'] - overlapping_speakers[j]['overlap_start']) / trans_duration)))
            for j in range(i)
        )
        end_word_idx = min(len(text_words), start_word_idx + word_count)
        
        speaker_words = text_words[start_word_idx:end_word_idx]
        speaker_text = " ".join(speaker_words)
        
        if speaker_text.strip():
            segments.append({
                "speaker": speaker_info['speaker'],
                "text": speaker_text.strip(),
                "start": speaker_start,
                "end": speaker_end,
                "duration": speaker_duration,
                "overlap_confidence": 0.8,
                "words": []
            })
    
    return segments


def align_segments(
    transcription: List[Dict], 
    diarization: List[Tuple[float, float, str]]
) -> List[Dict]:
    """
    Align transcription segments with speaker diarization using advanced splitting
    
    Args:
        transcription: List of transcription segments with start, end, text
        diarization: List of speaker segments (start, end, speaker)
        
    Returns:
        List of aligned segments with speaker, text, start, end
    """
    return align_segments_advanced(transcription, diarization)


def merge_consecutive_segments(aligned_segments: List[Dict], max_gap: float = 1.0) -> List[Dict]:
    """
    Merge consecutive segments from the same speaker
    
    Args:
        aligned_segments: List of aligned segments
        max_gap: Maximum gap in seconds to merge across
        
    Returns:
        List of merged segments
    """
    if not aligned_segments:
        return []
        
    merged = []
    current_segment = aligned_segments[0].copy()
    
    for segment in aligned_segments[1:]:
        # Check if we should merge with current segment
        should_merge = (
            segment["speaker"] == current_segment["speaker"] and
            segment["start"] - current_segment["end"] <= max_gap
        )
        
        if should_merge:
            # Merge segments
            current_segment["text"] += " " + segment["text"]
            current_segment["end"] = segment["end"]
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
            
            # Merge words if available
            if "words" in current_segment and "words" in segment:
                current_segment["words"].extend(segment["words"])
                
        else:
            # Start new segment
            merged.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    merged.append(current_segment)
    
    logger.info(f"Merged {len(aligned_segments)} segments into {len(merged)} segments")
    return merged


def create_speaker_timeline(aligned_segments: List[Dict]) -> Dict:
    """
    Create a summary timeline showing speaker activity
    
    Args:
        aligned_segments: List of aligned segments
        
    Returns:
        Dictionary with speaker statistics
    """
    speakers = {}
    total_duration = 0
    
    for segment in aligned_segments:
        speaker = segment["speaker"]
        duration = segment["duration"]
        
        if speaker not in speakers:
            speakers[speaker] = {
                "total_duration": 0,
                "segment_count": 0,
                "words": 0
            }
        
        speakers[speaker]["total_duration"] += duration
        speakers[speaker]["segment_count"] += 1
        
        # Count words
        text = segment["text"].strip()
        if text:
            speakers[speaker]["words"] += len(text.split())
        
        total_duration += duration
    
    # Calculate percentages
    for speaker_data in speakers.values():
        speaker_data["percentage"] = (
            speaker_data["total_duration"] / total_duration * 100 
            if total_duration > 0 else 0
        )
    
    return {
        "speakers": speakers,
        "total_duration": total_duration,
        "unique_speakers": len(speakers)
    } 
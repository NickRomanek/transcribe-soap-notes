"""
Main pipeline for AI Scribe - orchestrates transcription, diarization, alignment, and summarization
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import transcribe, diarize, align, summarise, speaker_roles, config

logger = logging.getLogger(__name__)
console = Console()


def create_speaker_dialogue_file(output_path: Path, aligned_segments: List[Dict[str, Any]], summary: Dict[str, Any] = None, role_info: Dict[str, Any] = None) -> Path:
    """
    Create a clean text file with speaker dialogue in readable format, with optional SOAP summary at top
    
    Args:
        output_path: Base output path
        aligned_segments: List of aligned segments with speaker info
        summary: Optional SOAP summary to include at top
        
    Returns:
        Path to the created dialogue file
    """
    dialogue_path = output_path.with_suffix('.txt')
    
    # Sort segments by start time for chronological order
    sorted_segments = sorted(aligned_segments, key=lambda x: x.get('start', 0))
    
    with open(dialogue_path, 'w', encoding='utf-8') as f:
        f.write("AI Scribe - Medical Transcription\n")
        f.write("=" * 50 + "\n\n")
        
        # Add SOAP summary at the top if available
        if summary and 'soap_note' in summary:
            f.write("SOAP NOTE SUMMARY\n")
            f.write("-" * 20 + "\n\n")
            
            soap_note = summary['soap_note']
            
            # Check if we have a raw_response with better formatting
            if 'raw_response' in summary and summary['raw_response']:
                try:
                    import json
                    # Try to parse the raw response as JSON
                    raw_data = json.loads(summary['raw_response'])
                    if 'soap_note' in raw_data:
                        soap_note = raw_data['soap_note']
                except:
                    # If parsing fails, use the original soap_note
                    pass
            
            f.write(f"SUBJECTIVE:\n{soap_note.get('subjective', 'N/A')}\n\n")
            f.write(f"OBJECTIVE:\n{soap_note.get('objective', 'N/A')}\n\n")
            f.write(f"ASSESSMENT:\n{soap_note.get('assessment', 'N/A')}\n\n")
            f.write(f"PLAN:\n{soap_note.get('plan', 'N/A')}\n\n")
            
            # Add confidence and key points if available
            if 'confidence' in summary:
                f.write(f"Confidence: {summary['confidence']:.0%}\n\n")
            
            if 'bullets' in summary and summary['bullets']:
                f.write("KEY POINTS:\n")
                for bullet in summary['bullets'][:5]:  # Limit to first 5 bullets
                    if isinstance(bullet, str) and len(bullet) < 200:  # Skip very long bullets
                        f.write(f"• {bullet}\n")
                f.write("\n")
            
            f.write("=" * 50 + "\n\n")
        
        f.write("SPEAKER DIALOGUE\n")
        f.write("-" * 20 + "\n\n")
        
        for segment in sorted_segments:
            # Format timestamp
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            timestamp = f"[{start_time:.1f}s - {end_time:.1f}s]"
            
            # Get speaker display name with role information
            speaker = segment.get('speaker', 'UNKNOWN')
            speaker_display = _get_speaker_display_name(speaker, role_info)
            
            # Write dialogue
            f.write(f"{speaker_display} {timestamp}:\n")
            f.write(f'"{segment.get("text", "").strip()}"\n\n')
    
    logger.info(f"Speaker dialogue saved to: {dialogue_path}")
    return dialogue_path


def _get_speaker_display_name(speaker: str, role_info: Dict[str, Any] = None) -> str:
    """
    Get a human-readable speaker display name based on role analysis
    
    Args:
        speaker: Original speaker ID (e.g., 'SPEAKER_00')
        role_info: Role analysis information
        
    Returns:
        Human-readable speaker name
    """
    if not role_info or speaker not in role_info:
        # Fallback to original format
        return f"Speaker {speaker.replace('SPEAKER_', '')}"
    
    role_data = role_info[speaker]
    role = role_data.get('role', 'unknown')
    confidence = role_data.get('confidence', 0)
    
    # Create role-based display names
    role_names = {
        'healthcare_provider': 'Healthcare Provider',
        'patient': 'Patient',
        'family_member': 'Family Member',
        'interpreter': 'Interpreter',
        'other': 'Other Participant'
    }
    
    base_name = role_names.get(role, 'Unknown Role')
    
    # Add confidence indicator if low confidence
    if confidence < 0.7:
        base_name += " (?)"
    
    # Add speaker ID for clarity
    speaker_num = speaker.replace('SPEAKER_', '')
    return f"{base_name} (Speaker {speaker_num})"


def _create_speaker_timeline(aligned_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create speaker timeline statistics"""
    speakers = {}
    total_duration = 0
    
    for segment in aligned_segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        if speaker not in speakers:
            speakers[speaker] = {
                'total_duration': 0,
                'segment_count': 0,
                'words': 0
            }
        
        speakers[speaker]['total_duration'] += duration
        speakers[speaker]['segment_count'] += 1
        speakers[speaker]['words'] += len(segment.get('text', '').split())
        total_duration += duration
    
    # Calculate percentages
    for speaker_data in speakers.values():
        speaker_data['percentage'] = (speaker_data['total_duration'] / total_duration * 100) if total_duration > 0 else 0
    
    return {
        'speakers': speakers,
        'total_duration': total_duration,
        'unique_speakers': len(speakers)
    }


def _create_statistics(original_segments: List[Dict], aligned_segments: List[Dict]) -> Dict[str, Any]:
    """Create processing statistics"""
    # Count merged segments (segments that were combined during alignment)
    merged_count = len([s for s in aligned_segments if len(s.get('words', [])) > 1])
    
    # Calculate total duration
    total_duration = 0
    if aligned_segments:
        total_duration = max(s.get('end', 0) for s in aligned_segments)
    
    return {
        'total_segments': len(original_segments),
        'aligned_segments': len(aligned_segments),
        'merged_segments': merged_count,
        'unique_speakers': len(set(s.get('speaker', 'UNKNOWN') for s in aligned_segments)),
        'total_duration_minutes': total_duration / 60
    }


def get_audio_duration(wav_path: Path) -> float:
    """Get audio duration in seconds"""
    try:
        # Try with wave module first (most reliable for WAV files)
        import wave
        with wave.open(str(wav_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception:
        try:
            # Fallback to librosa if available
            import librosa
            duration = librosa.get_duration(path=str(wav_path))
            return duration
        except ImportError:
            # If no audio libraries available, return 0
            return 0.0
        except Exception:
            return 0.0


def _create_short_segments_for_diarization(segments: List[Dict]) -> List[Tuple[float, float]]:
    """
    Create shorter time segments for more accurate diarization
    
    This addresses the issue where long transcription segments (3-5 sentences) 
    cause speaker misattribution during diarization. By creating shorter segments
    based on natural breaks, we get much better speaker identification.
    
    Args:
        segments: Original transcription segments
        
    Returns:
        List of (start_time, end_time) tuples for shorter segments
    """
    short_segments = []
    
    for segment in segments:
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time
        text = segment.get('text', '')
        words = segment.get('words', [])
        
        # If segment is short enough (< 3 seconds), keep as is
        if duration < 3.0:
            short_segments.append((start_time, end_time))
            continue
        
        # For longer segments, try to split at natural breaks
        if words:
            # Split based on word-level timestamps and punctuation
            current_start = start_time
            current_words = []
            
            for i, word in enumerate(words):
                word_text = word.get('word', '')
                word_end = word.get('end', current_start)
                current_words.append(word)
                
                # Check for natural break points
                is_punctuation = any(punct in word_text for punct in ['.', '!', '?', ',', ';'])
                is_pause = (i + 1 < len(words) and 
                           words[i + 1].get('start', word_end) - word_end > 0.3)  # 300ms pause
                segment_duration = word_end - current_start
                
                # Split if we hit a natural break and have enough content
                if (is_punctuation or is_pause) and segment_duration > 1.0:
                    short_segments.append((current_start, word_end))
                    current_start = word_end
                    current_words = []
                # Also split if segment gets too long (> 4 seconds)
                elif segment_duration > 4.0:
                    short_segments.append((current_start, word_end))
                    current_start = word_end
                    current_words = []
            
            # Add remaining words as final segment
            if current_words:
                short_segments.append((current_start, end_time))
        else:
            # Fallback: split long segments into 3-second chunks
            num_chunks = max(2, int(duration / 3.0))
            chunk_duration = duration / num_chunks
            
            for i in range(num_chunks):
                chunk_start = start_time + (i * chunk_duration)
                chunk_end = min(start_time + ((i + 1) * chunk_duration), end_time)
                short_segments.append((chunk_start, chunk_end))
    
    logger.info(f"Created {len(short_segments)} short segments from {len(segments)} transcription segments")
    return short_segments


def _refine_speaker_segments_with_short_segments(
    speaker_segments: List[Tuple[float, float, str]], 
    short_time_segments: List[Tuple[float, float]]
) -> List[Tuple[float, float, str]]:
    """
    Refine speaker segments by splitting them at natural break points
    
    This combines the diarization results with the shorter time segments
    to create more accurate speaker boundaries at natural speech breaks.
    
    Args:
        speaker_segments: Original speaker diarization results
        short_time_segments: Shorter time segments based on transcription
        
    Returns:
        Refined speaker segments with better boundaries
    """
    refined_segments = []
    
    for speaker_start, speaker_end, speaker in speaker_segments:
        # Find all short segments that overlap with this speaker segment
        overlapping_shorts = []
        for short_start, short_end in short_time_segments:
            # Check for overlap
            if short_end > speaker_start and short_start < speaker_end:
                # Clip to speaker boundaries
                clipped_start = max(short_start, speaker_start)
                clipped_end = min(short_end, speaker_end)
                if clipped_end > clipped_start:
                    overlapping_shorts.append((clipped_start, clipped_end))
        
        if overlapping_shorts:
            # Sort by start time
            overlapping_shorts.sort()
            
            # Create refined segments for each overlap
            for short_start, short_end in overlapping_shorts:
                refined_segments.append((short_start, short_end, speaker))
        else:
            # No overlapping short segments, keep original
            refined_segments.append((speaker_start, speaker_end, speaker))
    
    # Sort by start time and merge very close segments from same speaker
    refined_segments.sort()
    
    if not refined_segments:
        return speaker_segments
    
    merged_segments = []
    current_segment = list(refined_segments[0])
    
    for next_segment in refined_segments[1:]:
        next_start, next_end, next_speaker = next_segment
        curr_start, curr_end, curr_speaker = current_segment
        
        # Merge if same speaker and very close (< 0.2 seconds apart)
        if next_speaker == curr_speaker and next_start - curr_end < 0.2:
            current_segment[1] = next_end  # Extend end time
        else:
            merged_segments.append(tuple(current_segment))
            current_segment = list(next_segment)
    
    # Add the last segment
    merged_segments.append(tuple(current_segment))
    
    logger.info(f"Refined {len(speaker_segments)} speaker segments into {len(merged_segments)} segments using short segments")
    return merged_segments


def _post_process_speaker_consistency(aligned_segments: List[Dict]) -> List[Dict]:
    """
    Post-process aligned segments to fix speaker consistency issues
    
    This addresses cases where very short segments (< 1 second) create
    unnatural speaker switches in the middle of sentences or thoughts.
    
    Args:
        aligned_segments: List of aligned segments
        
    Returns:
        List of segments with improved speaker consistency
    """
    if len(aligned_segments) < 2:
        return aligned_segments
    
    # Sort by start time
    segments = sorted(aligned_segments, key=lambda x: x.get('start', 0))
    processed_segments = []
    i = 0
    
    while i < len(segments):
        current_segment = segments[i].copy()
        current_duration = current_segment.get('end', 0) - current_segment.get('start', 0)
        current_text = current_segment.get('text', '').strip()
        
        # Check if this is a very short segment that might be misattributed
        if current_duration < 1.0 and len(current_text) < 10:  # Short segment with little text
            # Look for nearby segments with the same speaker to merge with
            merged = False
            
            # Check previous segment
            if (processed_segments and 
                processed_segments[-1].get('speaker') == current_segment.get('speaker')):
                # Check if there's a small gap (< 0.5s) between segments
                prev_end = processed_segments[-1].get('end', 0)
                curr_start = current_segment.get('start', 0)
                
                if curr_start - prev_end < 0.5:
                    # Merge with previous segment
                    processed_segments[-1]['end'] = current_segment.get('end', 0)
                    processed_segments[-1]['text'] += ' ' + current_text
                    if 'words' in processed_segments[-1] and 'words' in current_segment:
                        processed_segments[-1]['words'].extend(current_segment.get('words', []))
                    merged = True
            
            # If not merged with previous, check next segment
            if not merged and i + 1 < len(segments):
                next_segment = segments[i + 1]
                next_start = next_segment.get('start', 0)
                curr_end = current_segment.get('end', 0)
                
                if (next_segment.get('speaker') == current_segment.get('speaker') and
                    next_start - curr_end < 0.5):
                    # Merge with next segment
                    current_segment['end'] = next_segment.get('end', 0)
                    current_segment['text'] += ' ' + next_segment.get('text', '').strip()
                    if 'words' in current_segment and 'words' in next_segment:
                        current_segment['words'].extend(next_segment.get('words', []))
                    processed_segments.append(current_segment)
                    i += 2  # Skip the next segment since we merged it
                    merged = True
            
            if not merged:
                # Check if we should reassign to adjacent speaker based on context
                reassigned = False
                
                # Check if previous and next segments have same speaker (sandwich case)
                if (processed_segments and i + 1 < len(segments) and
                    processed_segments[-1].get('speaker') == segments[i + 1].get('speaker') and
                    processed_segments[-1].get('speaker') != current_segment.get('speaker')):
                    
                    # Check gaps
                    prev_end = processed_segments[-1].get('end', 0)
                    curr_start = current_segment.get('start', 0)
                    curr_end = current_segment.get('end', 0)
                    next_start = segments[i + 1].get('start', 0)
                    
                    if (curr_start - prev_end < 0.3 and next_start - curr_end < 0.3):
                        # Reassign to the surrounding speaker
                        current_segment['speaker'] = processed_segments[-1].get('speaker')
                        reassigned = True
                
                processed_segments.append(current_segment)
        else:
            processed_segments.append(current_segment)
        
        i += 1
    
    logger.info(f"Post-processed speaker consistency: {len(aligned_segments)} → {len(processed_segments)} segments")
    return processed_segments


def run(wav_path: str, output_dir: Optional[Path] = None, progress_callback=None) -> Dict:
    """
    Run the complete AI Scribe pipeline on an audio file
    
    Args:
        wav_path: Path to WAV audio file
        output_dir: Output directory (optional)
        
    Returns:
        Dictionary containing all processing results
    """
    from datetime import datetime
    
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Get audio duration for time estimates
    audio_duration = get_audio_duration(wav_path)
    audio_mins = audio_duration / 60
    
    # Display initial info
    console.print(Panel(
        f"[bold blue]AI Scribe Pipeline[/bold blue]\n\n"
        f"📁 Input: {wav_path.name}\n"
        f"⏱️  Duration: {audio_mins:.1f} minutes\n"
        f"📂 Output: {output_dir}",
        title="Starting Processing",
        border_style="blue"
    ))
    
    start_time = datetime.now()
    
    # Create progress tracking
    with Progress(
        TextColumn("[bold blue]{task.fields[step]}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        
        # Add main progress task
        main_task = progress.add_task(
            "",
            total=6,  # Updated to include speaker analysis step
            step="🎯 AI Scribe Pipeline",
            status="Initializing..."
        )
        
        # Step 1: Transcribe
        progress.update(main_task, advance=1, status="🎤 Transcribing audio...")
        if progress_callback:
            progress_callback(16.7, "🎤 Transcribing audio...")
        step_start = time.time()
        
        try:
            segments = transcribe.transcribe(wav_path)
            step_time = time.time() - step_start
            console.print(f"✅ Transcription complete: {len(segments)} segments ({step_time:.1f}s)")
        except Exception as e:
            console.print(f"❌ Transcription failed: {e}")
            raise
        
        # Step 2: Create short segments and diarize
        progress.update(main_task, advance=1, status="👥 Identifying speakers...")
        if progress_callback:
            progress_callback(33.3, "👥 Identifying speakers...")
        step_start = time.time()
        
        try:
            # First create shorter segments for better diarization
            short_time_segments = _create_short_segments_for_diarization(segments)
            
            # Then run diarization on the audio file
            speaker_segments = diarize.diarize(wav_path)
            
            # Combine the approaches: use the shorter time segments to guide speaker assignment
            refined_speaker_segments = _refine_speaker_segments_with_short_segments(
                speaker_segments, short_time_segments
            )
            
            step_time = time.time() - step_start
            console.print(f"✅ Diarization complete: {len(refined_speaker_segments)} refined speaker segments ({step_time:.1f}s)")
        except Exception as e:
            console.print(f"❌ Diarization failed: {e}")
            raise
        
        # Step 3: Align
        progress.update(main_task, advance=1, status="🔗 Aligning speakers with text...")
        if progress_callback:
            progress_callback(50.0, "🔗 Aligning speakers with text...")
        step_start = time.time()
        
        try:
            aligned_segments = align.align_segments(segments, refined_speaker_segments)
            
            # Post-process for better speaker consistency
            aligned_segments = _post_process_speaker_consistency(aligned_segments)
            
            step_time = time.time() - step_start
            console.print(f"✅ Alignment complete: {len(aligned_segments)} aligned segments ({step_time:.1f}s)")
        except Exception as e:
            console.print(f"❌ Alignment failed: {e}")
            raise
        
        # Step 3.5: Analyze speaker roles and apply semantic corrections
        progress.update(main_task, status="🧠 Analyzing speaker roles...")
        if progress_callback:
            progress_callback(66.7, "🧠 Analyzing speaker roles...")
        step_start = time.time()
        
        try:
            speaker_analysis = speaker_roles.analyze_and_correct_speakers(aligned_segments)
            aligned_segments = speaker_analysis['segments']
            
            # Store the role analysis for later use
            role_assignments = speaker_analysis['final_roles']
            corrections_made = speaker_analysis['corrected_segments'] - speaker_analysis['original_segments']
            
            step_time = time.time() - step_start
            console.print(f"✅ Speaker analysis complete: {corrections_made} corrections applied ({step_time:.1f}s)")
        except Exception as e:
            step_time = time.time() - step_start
            console.print(f"⚠️  Speaker analysis failed: {e} ({step_time:.1f}s)")
            role_assignments = {}
            speaker_analysis = {}
        
        # Step 4: Summarize
        progress.update(main_task, advance=1, status="📝 Generating summary...")
        if progress_callback:
            progress_callback(83.3, "📝 Generating SOAP summary...")
        step_start = time.time()
        
        try:
            summary = summarise.summarize(aligned_segments)
            step_time = time.time() - step_start
            console.print(f"✅ Summary complete ({step_time:.1f}s)")
        except Exception as e:
            step_time = time.time() - step_start
            console.print(f"⚠️  Summary failed: {e} ({step_time:.1f}s)")
            summary = {
                "error": f"Summarization failed: {e}",
                "soap_note": {
                    "subjective": "Unable to generate",
                    "objective": "Unable to generate", 
                    "assessment": "Unable to generate",
                    "plan": "Unable to generate"
                },
                "bullets": ["Summarization failed"],
                "confidence": 0.0
            }
        
        # Step 5: Create speaker dialogue file
        progress.update(main_task, advance=1, status="💬 Creating dialogue file...")
        if progress_callback:
            progress_callback(95.0, "💬 Creating dialogue file...")
        step_start = time.time()
        
        try:
            # Pass role assignments to dialogue file creation
            role_info = role_assignments if 'role_assignments' in locals() else {}
            dialogue_path = create_speaker_dialogue_file(output_dir / wav_path.stem, aligned_segments, summary, role_info)
            step_time = time.time() - step_start
            console.print(f"✅ Dialogue file created: {dialogue_path.name} ({step_time:.1f}s)")
        except Exception as e:
            step_time = time.time() - step_start
            console.print(f"❌ Dialogue file creation failed: {e} ({step_time:.1f}s)")
            dialogue_path = None
        
        # Complete progress
        progress.update(main_task, status="✅ Complete!")
        
        # Compile results
        speaker_timeline = _create_speaker_timeline(aligned_segments)
        statistics = _create_statistics(segments, aligned_segments)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results = {
            "input_file": str(wav_path),
            "output_directory": str(output_dir),
            "processing_time_seconds": processing_time,
            "aligned_segments": aligned_segments,
            "speaker_timeline": speaker_timeline,
            "speaker_analysis": speaker_analysis if 'speaker_analysis' in locals() else {},
            "soap_summary": summary,
            "statistics": statistics,
            "dialogue_file": str(dialogue_path) if dialogue_path else None
        }
        
        # Save results
        try:
            output_file = output_dir / f"{wav_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            console.print(f"💾 Results saved to: {output_file.name}")
        except Exception as e:
            console.print(f"❌ Failed to save results: {e}")
            raise
        
        # Display final summary
        console.print(Panel(
            f"[bold green]Processing Complete![/bold green]\n\n"
            f"⏱️  Total time: {processing_time:.1f} seconds\n"
            f"📊 Segments: {len(segments)} → {len(aligned_segments)} aligned\n"
            f"👥 Speakers: {statistics['unique_speakers']}\n"
            f"📁 Files created:\n"
            f"   • {output_file.name}\n"
            f"   • {dialogue_path.name if dialogue_path else 'dialogue file failed'}",
            title="✅ Success",
            border_style="green"
        ))
        
        return results


def validate_audio_file(wav_path: Path) -> bool:
    """
    Validate that the audio file is suitable for processing
    
    Args:
        wav_path: Path to audio file
        
    Returns:
        True if valid, raises exception if invalid
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
    if wav_path.suffix.lower() not in ['.wav', '.mp3', '.m4a', '.flac']:
        raise ValueError(f"Unsupported audio format: {wav_path.suffix}")
        
    # Check file size (reasonable limits)
    file_size_mb = wav_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:  # 500MB limit
        raise ValueError(f"Audio file too large: {file_size_mb:.1f}MB")
        
    return True


def list_recent_results(output_dir: Optional[Path] = None, limit: int = 10) -> List[Dict]:
    """
    List recent processing results
    
    Args:
        output_dir: Output directory to search
        limit: Maximum number of results to return
        
    Returns:
        List of result summaries
    """
    output_dir = output_dir or config.OUTPUT_DIR
    
    if not output_dir.exists():
        return []
        
    json_files = list(output_dir.glob("*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    results = []
    for json_file in json_files[:limit]:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            summary = {
                "filename": json_file.name,
                "input_file": data.get("metadata", {}).get("input_file", "unknown"),
                "timestamp": data.get("metadata", {}).get("timestamp", "unknown"),
                "processing_time": data.get("metadata", {}).get("processing_time_seconds", 0),
                "speakers": data.get("statistics", {}).get("unique_speakers", 0),
                "duration_minutes": data.get("statistics", {}).get("total_duration_minutes", 0),
                "has_error": data.get("metadata", {}).get("error", False)
            }
            results.append(summary)
            
        except Exception as e:
            logger.warning(f"Failed to read {json_file}: {e}")
            
    return results 
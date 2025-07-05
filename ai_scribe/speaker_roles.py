"""
Speaker role detection and semantic correction using LLM analysis
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

# Try to import LLM dependencies
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class SpeakerRoleAnalyzer:
    """
    Analyzes conversation context to identify speaker roles and correct misattributions
    """
    
    def __init__(self):
        self.llm = None
        # Use separate model for role analysis if configured
        if config.USE_SEPARATE_ROLE_MODEL and config.ROLE_ANALYSIS_MODEL.exists():
            self.model_path = config.ROLE_ANALYSIS_MODEL
            self.model_config = config.ROLE_MODEL_CONFIG
            logger.info("Using separate model for role analysis")
        else:
            self.model_path = config.LLM_MODEL
            self.model_config = config.SUMMARY_MODEL_CONFIG
            logger.info("Using summary model for role analysis")
        
    def load_model(self):
        """Load the LLM model for speaker analysis"""
        if Llama is None:
            logger.warning("llama-cpp-python not available, using heuristic analysis only")
            return
            
        if not self.model_path.exists():
            logger.warning(f"LLM model not found: {self.model_path}, using heuristic analysis only")
            return
            
        try:
            logger.info(f"Loading LLM model for speaker analysis: {self.model_path}")
            
            # Configure based on available resources
            n_gpu_layers = -1 if config.DEVICE == "cuda" else 0
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.model_config["n_ctx"],
                n_batch=self.model_config["n_batch"],
                verbose=False
            )
            
            logger.info("LLM model loaded successfully for speaker analysis")
            
        except Exception as e:
            logger.warning(f"Failed to load LLM model: {e}, using heuristic analysis only")
            self.llm = None
    
    def analyze_speaker_roles(self, aligned_segments: List[Dict]) -> Dict[str, Any]:
        """
        Analyze conversation to identify speaker roles and suggest corrections
        
        Args:
            aligned_segments: List of aligned segments with speaker info
            
        Returns:
            Dictionary with speaker role analysis and corrections
        """
        logger.info("Analyzing speaker roles and conversation context")
        
        # Phase 1: Heuristic analysis (always available)
        heuristic_analysis = self._heuristic_role_detection(aligned_segments)
        
        # Phase 2: LLM analysis (if available)
        llm_analysis = None
        if self.llm is None:
            self.load_model()
            
        if self.llm is not None:
            llm_analysis = self._llm_role_analysis(aligned_segments)
        
        # Phase 3: Combine analyses and apply corrections
        final_analysis = self._combine_analyses(heuristic_analysis, llm_analysis)
        corrected_segments = self._apply_semantic_corrections(aligned_segments, final_analysis)
        
        return {
            'original_segments': len(aligned_segments),
            'corrected_segments': len(corrected_segments),
            'heuristic_analysis': heuristic_analysis,
            'llm_analysis': llm_analysis,
            'final_roles': final_analysis,
            'segments': corrected_segments
        }
    
    def _heuristic_role_detection(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Use heuristic patterns to detect speaker roles
        """
        logger.info("Running heuristic speaker role detection")
        
        speaker_stats = {}
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').lower()
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'questions_asked': 0,
                    'medical_terms': 0,
                    'symptom_descriptions': 0,
                    'total_words': 0,
                    'segments': 0,
                    'explanations_given': 0
                }
            
            stats = speaker_stats[speaker]
            stats['segments'] += 1
            stats['total_words'] += len(text.split())
            
            # Count questions
            if '?' in text or any(q in text for q in ['what', 'how', 'when', 'where', 'why', 'can you', 'do you']):
                stats['questions_asked'] += 1
            
            # Count medical terms
            medical_terms = [
                'blood pressure', 'heart rate', 'temperature', 'symptoms', 'diagnosis', 
                'medication', 'prescription', 'treatment', 'examination', 'patient',
                'doctor', 'nurse', 'clinic', 'hospital', 'pain', 'chest', 'breathing',
                'headache', 'fever', 'nausea', 'dizzy', 'allergic', 'infection'
            ]
            stats['medical_terms'] += sum(1 for term in medical_terms if term in text)
            
            # Count symptom descriptions
            symptom_patterns = [
                'i feel', 'i have', 'it hurts', 'pain in', 'been having', 'started',
                'getting worse', 'feeling', 'experiencing', 'bothering me'
            ]
            stats['symptom_descriptions'] += sum(1 for pattern in symptom_patterns if pattern in text)
            
            # Count explanations
            explanation_patterns = [
                'this means', 'what happens', 'the reason', 'because', 'due to',
                'caused by', 'let me explain', 'basically', 'in other words'
            ]
            stats['explanations_given'] += sum(1 for pattern in explanation_patterns if pattern in text)
        
        # Analyze patterns to assign roles
        role_assignments = {}
        speakers = list(speaker_stats.keys())
        
        for speaker in speakers:
            stats = speaker_stats[speaker]
            
            # Calculate ratios
            question_ratio = stats['questions_asked'] / max(stats['segments'], 1)
            medical_ratio = stats['medical_terms'] / max(stats['total_words'], 1)
            symptom_ratio = stats['symptom_descriptions'] / max(stats['segments'], 1)
            explanation_ratio = stats['explanations_given'] / max(stats['segments'], 1)
            
            # Determine role based on patterns
            provider_score = question_ratio * 2 + medical_ratio * 10 + explanation_ratio * 3
            patient_score = symptom_ratio * 5 + (1 - question_ratio) * 2
            
            if provider_score > patient_score:
                role = 'healthcare_provider'
                confidence = min(0.95, provider_score / (provider_score + patient_score))
            else:
                role = 'patient'
                confidence = min(0.95, patient_score / (provider_score + patient_score))
            
            role_assignments[speaker] = {
                'role': role,
                'confidence': confidence,
                'evidence': {
                    'questions_asked': stats['questions_asked'],
                    'medical_terms': stats['medical_terms'],
                    'symptom_descriptions': stats['symptom_descriptions'],
                    'explanations_given': stats['explanations_given']
                }
            }
        
        logger.info(f"Heuristic analysis complete: {role_assignments}")
        return role_assignments
    
    def _llm_role_analysis(self, segments: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze conversation context and identify speaker roles
        """
        logger.info("Running LLM-based speaker role analysis")
        
        try:
            # Format conversation for LLM analysis
            conversation = self._format_conversation_for_llm(segments)
            
            prompt = f"""<|system|>
You are a medical conversation analyst. Analyze the following conversation between speakers and identify their roles.

<|user|>
Analyze this medical conversation and identify the role of each speaker:

{conversation}

Based on the conversation patterns, medical terminology usage, question-asking behavior, and content:

1. Who is the healthcare provider (doctor, nurse, medical professional)?
2. Who is the patient?
3. Are there any other participants (family members, interpreters)?

Return your analysis in JSON format:
{{
    "SPEAKER_00": {{
        "role": "healthcare_provider|patient|family_member|interpreter|other",
        "confidence": 0.95,
        "evidence": ["specific evidence from conversation"]
    }},
    "SPEAKER_01": {{
        "role": "healthcare_provider|patient|family_member|interpreter|other",
        "confidence": 0.90,
        "evidence": ["specific evidence from conversation"]
    }}
}}

<|assistant|>"""

            # Generate response
            response = self.llm(
                prompt,
                max_tokens=512,  # Shorter for role analysis
                temperature=self.model_config["temperature"],
                top_p=0.9,
                stop=["<|user|>", "<|system|>"]
            )
            
            response_text = response['choices'][0]['text'].strip()
            logger.info(f"LLM raw response: {response_text}")
            
            # Try to parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    analysis = json.loads(json_str)
                    logger.info(f"LLM analysis successful: {analysis}")
                    return analysis
                else:
                    logger.warning("No JSON found in LLM response")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                return None
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
    
    def _format_conversation_for_llm(self, segments: List[Dict]) -> str:
        """Format conversation for LLM analysis"""
        formatted_lines = []
        
        for segment in segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            
            formatted_lines.append(f"{speaker} [{start_time:.1f}s]: \"{text}\"")
        
        return "\n".join(formatted_lines)
    
    def _combine_analyses(self, heuristic: Dict, llm: Optional[Dict]) -> Dict[str, Any]:
        """
        Combine heuristic and LLM analyses to determine final speaker roles
        """
        logger.info("Combining heuristic and LLM analyses")
        
        final_roles = {}
        
        for speaker in heuristic.keys():
            heuristic_role = heuristic[speaker]
            
            if llm and speaker in llm:
                llm_role = llm[speaker]
                
                # If both analyses agree, use higher confidence
                if heuristic_role['role'] == llm_role['role']:
                    confidence = max(heuristic_role['confidence'], llm_role['confidence'])
                    evidence = heuristic_role['evidence']
                    if 'evidence' in llm_role:
                        evidence['llm_evidence'] = llm_role['evidence']
                else:
                    # If they disagree, prefer LLM but lower confidence
                    confidence = llm_role['confidence'] * 0.8
                    evidence = {
                        'heuristic': heuristic_role,
                        'llm': llm_role,
                        'note': 'analyses_disagreed'
                    }
                
                final_roles[speaker] = {
                    'role': llm_role['role'],
                    'confidence': confidence,
                    'evidence': evidence,
                    'analysis_method': 'combined'
                }
            else:
                # Use heuristic only
                final_roles[speaker] = {
                    **heuristic_role,
                    'analysis_method': 'heuristic_only'
                }
        
        logger.info(f"Final role assignments: {final_roles}")
        return final_roles
    
    def _apply_semantic_corrections(self, segments: List[Dict], role_analysis: Dict) -> List[Dict]:
        """
        Apply semantic corrections based on role analysis
        """
        logger.info("Applying semantic corrections to speaker assignments")
        
        corrected_segments = []
        corrections_made = 0
        
        for segment in segments:
            corrected_segment = segment.copy()
            original_speaker = segment.get('speaker', 'UNKNOWN')
            text = segment.get('text', '').lower()
            
            # Check for obvious misattributions based on content
            corrections = self._detect_misattributions(text, original_speaker, role_analysis)
            
            if corrections:
                corrected_segment['speaker'] = corrections['suggested_speaker']
                corrected_segment['correction_reason'] = corrections['reason']
                corrected_segment['original_speaker'] = original_speaker
                corrections_made += 1
                
                logger.debug(f"Corrected speaker: {original_speaker} -> {corrections['suggested_speaker']} "
                           f"for text: {text[:50]}... Reason: {corrections['reason']}")
            
            corrected_segments.append(corrected_segment)
        
        logger.info(f"Applied {corrections_made} semantic corrections")
        return corrected_segments
    
    def _detect_misattributions(self, text: str, current_speaker: str, role_analysis: Dict) -> Optional[Dict]:
        """
        Detect obvious speaker misattributions based on content
        """
        # Find healthcare provider and patient speakers
        provider_speaker = None
        patient_speaker = None
        
        for speaker, analysis in role_analysis.items():
            if analysis['role'] == 'healthcare_provider':
                provider_speaker = speaker
            elif analysis['role'] == 'patient':
                patient_speaker = speaker
        
        if not provider_speaker or not patient_speaker:
            return None
        
        # Check for provider-specific content assigned to patient
        if (current_speaker == patient_speaker and 
            any(phrase in text for phrase in [
                'let me examine', 'i need to check', 'your blood pressure',
                'take your temperature', 'what brings you in', 'how long has',
                'any allergies', 'current medications', 'let me explain',
                'the diagnosis is', 'i recommend', 'prescription for'
            ])):
            return {
                'suggested_speaker': provider_speaker,
                'reason': 'provider_content_detected'
            }
        
        # Check for patient-specific content assigned to provider
        if (current_speaker == provider_speaker and 
            any(phrase in text for phrase in [
                'i feel', 'i have been', 'it hurts when', 'the pain started',
                'i noticed', 'i am experiencing', 'it bothers me',
                'i take', 'i am allergic to', 'my symptoms'
            ])):
            return {
                'suggested_speaker': patient_speaker,
                'reason': 'patient_content_detected'
            }
        
        return None


def analyze_and_correct_speakers(aligned_segments: List[Dict]) -> Dict[str, Any]:
    """
    Main function to analyze speaker roles and apply corrections
    
    Args:
        aligned_segments: List of aligned segments with speaker info
        
    Returns:
        Dictionary with analysis results and corrected segments
    """
    analyzer = SpeakerRoleAnalyzer()
    return analyzer.analyze_speaker_roles(aligned_segments) 
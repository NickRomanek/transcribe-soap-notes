"""
Summarization module using llama-cpp-python with Phi-3
"""

from typing import Dict, List, Optional
from pathlib import Path
import logging
import json

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from . import config

logger = logging.getLogger(__name__)


class Summarizer:
    """LLM-based summarization using llama-cpp"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.LLM_MODEL
        self.llm = None
        
    def load_model(self):
        """Load the LLM model"""
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed")
            
        if not self.model_path.exists():
            logger.warning(f"LLM model not found: {self.model_path}")
            logger.info("Summarization will use fallback method without LLM")
            return
            
        logger.info(f"Loading LLM model from {self.model_path}")
        
        # Configure based on available resources
        n_gpu_layers = -1 if config.DEVICE == "cuda" else 0
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,  # Context length for Phi-3 mini 4k
            n_batch=512,
            verbose=False
        )
        
        logger.info("LLM model loaded successfully")
        
    def create_prompt(self, aligned_segments: List[Dict]) -> str:
        """
        Create prompt for SOAP note generation
        
        Args:
            aligned_segments: List of aligned transcript segments
            
        Returns:
            Formatted prompt string
        """
        # Format transcript with speakers
        transcript_text = ""
        for segment in aligned_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            if text:
                transcript_text += f"{speaker}: {text}\n"
        
        prompt = f"""<|system|>
You are a medical assistant specializing in creating SOAP notes from patient consultations. You must analyze the provided transcript and create a comprehensive SOAP note following standard medical documentation practices.

SOAP Format:
- Subjective: Patient's reported symptoms, concerns, and history
- Objective: Observable data, vital signs, examination findings
- Assessment: Clinical impression, diagnoses, analysis
- Plan: Treatment recommendations, follow-up, medications

Instructions:
1. Extract relevant medical information from all speakers
2. Identify who is the healthcare provider vs patient/family
3. Focus on medically relevant content only
4. Use professional medical terminology
5. Organize information clearly in SOAP format
6. Return your response as a JSON object with the structure shown below

Required JSON Response Format:
{{
    "soap_note": {{
        "subjective": "Patient's reported symptoms and history...",
        "objective": "Physical examination findings and vital signs...",
        "assessment": "Clinical impressions and diagnoses...",
        "plan": "Treatment plan and recommendations..."
    }},
    "bullets": [
        "Key point 1",
        "Key point 2",
        "Key point 3"
    ],
    "confidence": 0.85,
    "participants": {{
        "provider": "SPEAKER_ID",
        "patient": "SPEAKER_ID"
    }}
}}
<|end|>

<|user|>
Please analyze the following medical consultation transcript and create a SOAP note:

{transcript_text}
<|end|>

<|assistant|>"""
        
        return prompt
        
    def summarize(self, aligned_segments: List[Dict]) -> Dict:
        """
        Generate SOAP note summary from aligned segments
        
        Args:
            aligned_segments: List of aligned transcript segments
            
        Returns:
            Dictionary with soap_note, bullets, confidence
        """
        if self.llm is None:
            self.load_model()
            
        # If no LLM model available, use fallback
        if self.llm is None:
            logger.info("No LLM available, generating basic summary")
            return self.generate_clinical_summary(aligned_segments)
            
        logger.info("Generating SOAP note summary")
        
        prompt = self.create_prompt(aligned_segments)
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            stop=["<|end|>", "<|user|>"],
            echo=False
        )
        
        response_text = response["choices"][0]["text"].strip()
        
        # Try to parse JSON response
        try:
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["soap_note", "bullets", "confidence"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in response")
                
            # Ensure SOAP note has required sections
            soap_sections = ["subjective", "objective", "assessment", "plan"]
            if not all(section in result["soap_note"] for section in soap_sections):
                logger.warning("SOAP note missing some sections")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback: create structured response from raw text
            result = {
                "soap_note": {
                    "subjective": "Unable to parse structured response",
                    "objective": "Unable to parse structured response", 
                    "assessment": "Unable to parse structured response",
                    "plan": "Unable to parse structured response"
                },
                "bullets": [response_text[:200] + "..." if len(response_text) > 200 else response_text],
                "confidence": 0.3,
                "raw_response": response_text
            }
        
        logger.info("SOAP note generation complete")
        return result
        
    def generate_clinical_summary(self, aligned_segments: List[Dict]) -> Dict:
        """
        Generate a brief clinical summary (alternative to full SOAP)
        
        Args:
            aligned_segments: List of aligned transcript segments
            
        Returns:
            Dictionary with summary information
        """
        # Extract basic statistics
        total_duration = sum(seg.get("duration", 0) for seg in aligned_segments)
        speakers = list(set(seg.get("speaker", "UNKNOWN") for seg in aligned_segments))
        total_words = sum(len(seg.get("text", "").split()) for seg in aligned_segments)
        
        # Create basic summary
        transcript_text = " ".join(seg.get("text", "") for seg in aligned_segments)
        
        return {
            "duration_minutes": round(total_duration / 60, 1),
            "speakers": speakers,
            "word_count": total_words,
            "key_topics": self._extract_key_topics(transcript_text),
            "summary": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text
        }
        
    def _extract_key_topics(self, text: str) -> List[str]:
        """Simple keyword extraction for key topics"""
        # This is a simple implementation - could be enhanced with NLP
        medical_keywords = [
            "pain", "medication", "treatment", "symptoms", "diagnosis",
            "therapy", "surgery", "blood pressure", "temperature", "history"
        ]
        
        text_lower = text.lower()
        found_topics = [keyword for keyword in medical_keywords if keyword in text_lower]
        return found_topics[:5]  # Return top 5


def summarize(aligned_segments: List[Dict]) -> Dict:
    """
    Convenience function for summarization
    
    Args:
        aligned_segments: List of aligned transcript segments
        
    Returns:
        SOAP note summary
    """
    summarizer = Summarizer()
    return summarizer.summarize(aligned_segments) 
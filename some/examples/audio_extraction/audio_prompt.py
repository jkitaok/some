from __future__ import annotations

from typing import Any, Dict
import os

from .audio_schema import (
    AudioAnalysis, AudioEvaluation, AudioSummary,
    MeetingAnalysis, PodcastAnalysis
)
from some.prompting import BasePromptBuilder
from some.media import get_audio_info, is_valid_media_url

class AudioAnalysisPrompt(BasePromptBuilder):
    """Prompt builder for comprehensive audio analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for audio analysis.
        
        Args:
            item: Dict containing:
                - audio_path: str - Path to audio file (local path or URL)
                - text: Optional[str] - Text context
                - context: Optional[str] - Additional context about the audio
                - expected_type: Optional[str] - Expected audio type for better analysis
        """
        audio_path = item.get("audio_path")
        text = item.get("text", "")
        context = item.get("context", "")
        expected_type = item.get("expected_type", "")

        if not audio_path:
            raise ValueError("audio_path is required for audio analysis")

        # Validate audio source using media.py functions
        if audio_path.startswith(('http://', 'https://')):
            # It's a URL
            if not is_valid_media_url(audio_path):
                raise ValueError(f"Invalid audio URL: {audio_path}")
        else:
            # It's a local file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            # Get audio info for better prompting
            try:
                audio_info = get_audio_info(audio_path)
                duration = audio_info.get('duration_seconds')
                if duration:
                    context += f"\nAudio duration: {duration:.1f} seconds"
            except Exception:
                pass  # Continue without audio info if it fails
        
        # Build comprehensive analysis prompt
        prompt_text = """Analyze this audio file and extract comprehensive information following the AudioAnalysis schema.

ANALYSIS GUIDELINES:
1. **Listen Carefully**: Pay attention to all speakers, background sounds, and audio quality
2. **Complete Transcript**: Provide a full, accurate transcript of all spoken content
3. **Speaker Analysis**: Identify and analyze each distinct speaker
4. **Content Extraction**: Extract key topics, names, places, and organizations mentioned
5. **Quality Assessment**: Evaluate audio quality and note any issues
6. **Confidence Scoring**: Provide honest confidence based on audio clarity and completeness

WHAT TO ANALYZE:
- **Content**: Full transcript, main topics, key points
- **Speakers**: Number of speakers, their characteristics, speaking styles
- **Entities**: Names, places, organizations mentioned
- **Technical**: Audio quality, background noise, clarity issues
- **Context**: Type of audio (meeting, podcast, interview, etc.)
- **Sentiment**: Overall tone and emotional content

TRANSCRIPT REQUIREMENTS:
- Include ALL spoken words, even if unclear (mark as [unclear] or [inaudible])
- Use speaker labels (Speaker 1, Speaker 2, etc.) for multi-speaker content
- Include significant pauses, interruptions, or overlapping speech
- Note any non-speech sounds that are relevant [laughter], [applause], etc.

CONFIDENCE SCORING:
- 0.9-1.0: Crystal clear audio, perfect understanding
- 0.7-0.8: Good quality, minor unclear sections
- 0.5-0.6: Moderate quality, some difficult sections
- 0.3-0.4: Poor quality but basic content extractable
- 0.0-0.2: Very poor quality, mostly guessing

AUDIO QUALITY ASSESSMENT:
- "excellent": Studio quality, no issues
- "good": Clear with minor background noise
- "fair": Understandable but some quality issues
- "poor": Difficult to understand, significant issues"""

        if text:
            prompt_text += f"\n\nTEXT CONTEXT:\n{text}"

        if expected_type:
            prompt_text += f"\n\nEXPECTED AUDIO TYPE: {expected_type}"

        if context:
            prompt_text += f"\n\nADDITIONAL CONTEXT:\n{context}"
        
        prompt_text += "\n\nAnalyze the audio and provide results following the AudioAnalysis schema exactly."
        
        return {
            "prompt_text": prompt_text,
            "audio_path": audio_path,
            "response_format": AudioAnalysis,
            "result_key": "audio_analysis",
        }

class AudioSummaryPrompt(BasePromptBuilder):
    """Prompt builder for quick audio summarization."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for audio summarization."""
        audio_path = item.get("audio_path")

        if not audio_path:
            raise ValueError("audio_path is required for audio summarization")
        
        prompt_text = """Provide a quick summary and transcript of this audio file.

Focus on:
1. **Brief Summary**: What is this audio about in 2-3 sentences
2. **Complete Transcript**: Full transcript of all spoken content
3. **Main Topic**: Primary subject or theme
4. **Speaker Count**: How many distinct speakers
5. **Audio Type**: What type of audio content this is
6. **Confidence**: Your confidence in the analysis

Keep the summary concise but ensure the transcript is complete and accurate."""
        
        return {
            "prompt_text": prompt_text,
            "audio_path": audio_path,
            "response_format": AudioSummary,
            "result_key": "audio_summary",
        }

class MeetingAnalysisPrompt(BasePromptBuilder):
    """Prompt builder for meeting audio analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for meeting analysis."""
        audio_path = item.get("audio_path")
        meeting_context = item.get("meeting_context", "")

        if not audio_path:
            raise ValueError("audio_path is required for meeting analysis")
        
        prompt_text = """Analyze this meeting audio and extract structured meeting information.

MEETING ANALYSIS FOCUS:
1. **Meeting Type**: Identify the type of meeting (standup, planning, review, etc.)
2. **Participants**: Extract names of participants if mentioned
3. **Agenda Items**: What topics were discussed
4. **Action Items**: Tasks or actions assigned to people
5. **Decisions**: What decisions were made
6. **Next Steps**: Follow-up actions or next meetings
7. **Full Transcript**: Complete transcript with speaker identification

EXTRACTION GUIDELINES:
- Listen for participant names and roles
- Identify specific action items and who they're assigned to
- Note any deadlines or timelines mentioned
- Extract key decisions and their rationale
- Identify follow-up meetings or check-ins planned"""

        if meeting_context:
            prompt_text += f"\n\nMEETING CONTEXT:\n{meeting_context}"
        
        prompt_text += "\n\nProvide meeting analysis following the MeetingAnalysis schema."
        
        return {
            "prompt_text": prompt_text,
            "audio_path": audio_path,
            "response_format": MeetingAnalysis,
            "result_key": "meeting_analysis",
        }

class PodcastAnalysisPrompt(BasePromptBuilder):
    """Prompt builder for podcast audio analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for podcast analysis."""
        audio_path = item.get("audio_path")

        if not audio_path:
            raise ValueError("audio_path is required for podcast analysis")
        
        prompt_text = """Analyze this podcast audio and extract podcast-specific information.

PODCAST ANALYSIS FOCUS:
1. **Show Information**: Podcast name, episode title if mentioned
2. **Host & Guests**: Identify hosts and any guests
3. **Episode Content**: Summary of what was discussed
4. **Key Quotes**: Notable or quotable moments
5. **Topics**: Main topics covered in the episode
6. **Sponsors**: Any sponsor mentions or advertisements
7. **Full Transcript**: Complete transcript with speaker identification

EXTRACTION GUIDELINES:
- Listen for podcast intro/outro with show name
- Identify host names and guest introductions
- Note sponsor reads or advertisement segments
- Extract memorable quotes or key insights
- Identify main topics and themes discussed
- Pay attention to episode structure and segments"""
        
        prompt_text += "\n\nProvide podcast analysis following the PodcastAnalysis schema."
        
        return {
            "prompt_text": prompt_text,
            "audio_path": audio_path,
            "response_format": PodcastAnalysis,
            "result_key": "podcast_analysis",
        }

class AudioEvaluationPrompt(BasePromptBuilder):
    """Prompt builder for evaluating audio extraction quality."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for audio extraction evaluation."""
        audio_path = item.get("audio_path")
        extraction_result = item["extraction_result"]
        expected_content = item.get("expected_content")

        if not audio_path:
            raise ValueError("audio_path is required for audio evaluation")
        
        prompt_text = """Evaluate the quality of this audio extraction by listening to the original audio and comparing it with the extracted analysis.

EVALUATION CRITERIA:
1. **Transcript Accuracy**: Is the transcript accurate and complete?
2. **Speaker Identification**: Are speakers correctly identified and characterized?
3. **Content Understanding**: Is the content properly understood and summarized?
4. **Information Extraction**: Are key details (names, topics, etc.) correctly extracted?
5. **Schema Compliance**: Does the output properly follow the required schema?
6. **Confidence Assessment**: Is the confidence score appropriate for the audio quality?

ASSESSMENT GUIDELINES:
- **Excellent**: Perfect transcript, all speakers identified, complete analysis
- **Good**: Minor transcript errors, most content correctly extracted
- **Fair**: Some transcript issues but main content understood
- **Poor**: Significant errors in transcript or content understanding

WHAT TO CHECK:
- Transcript completeness and accuracy
- Speaker count and identification
- Key information extraction (names, topics, decisions)
- Audio quality assessment accuracy
- Appropriate confidence scoring"""

        if expected_content:
            prompt_text += f"\n\nEXPECTED CONTENT (for reference):\n{expected_content}"
        
        prompt_text += f"""

EXTRACTION RESULT TO EVALUATE:
{extraction_result}

Provide evaluation following the AudioEvaluation schema."""
        
        return {
            "prompt_text": prompt_text,
            "audio_path": audio_path,
            "response_format": AudioEvaluation,
            "result_key": "evaluation",
        }

from __future__ import annotations

from typing import Any, Dict
import os

from .multimodal_schema import (
    MultiModalAnalysis, TextOnlyAnalysis, VisionOnlyAnalysis, AudioOnlyAnalysis,
    TextVisionAnalysis, TextAudioAnalysis, VisionAudioAnalysis
)
from some.prompting import BasePromptBuilder
from some.media import get_image_info, get_audio_info, is_valid_media_url

class MultiModalPrompt(BasePromptBuilder):
    """Prompt builder for comprehensive multi-modal analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for multi-modal analysis.
        
        Args:
            item: Dict containing:
                - text: Optional[str] - Text content to analyze
                - image_path: Optional[str] - Path to image file
                - image_url: Optional[str] - URL to image file
                - audio_path: Optional[str] - Path to audio file
                - audio_url: Optional[str] - URL to audio file
                - context: Optional[str] - Additional context
        """
        text = item.get("text", "")
        image_path = item.get("image_path")
        image_url = item.get("image_url")
        audio_path = item.get("audio_path")
        audio_url = item.get("audio_url")
        context = item.get("context", "")
        
        # Determine available modalities and validate sources
        has_text = bool(text.strip())
        has_image = bool(image_path or image_url)
        has_audio = bool(audio_path or audio_url)

        # Validate and enhance context with media info
        media_context = ""

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                image_info = get_image_info(image_path)
                dimensions = f"{image_info.get('width', 'unknown')}x{image_info.get('height', 'unknown')}"
                media_context += f"\nImage: {dimensions} pixels"
            except Exception:
                pass
        elif image_url:
            if not is_valid_media_url(image_url):
                raise ValueError(f"Invalid image URL: {image_url}")

        if audio_path:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            try:
                audio_info = get_audio_info(audio_path)
                duration = audio_info.get('duration_seconds')
                if duration:
                    media_context += f"\nAudio: {duration:.1f} seconds duration"
            except Exception:
                pass
        elif audio_url:
            if not is_valid_media_url(audio_url):
                raise ValueError(f"Invalid audio URL: {audio_url}")
        
        # Build comprehensive multi-modal prompt
        base_prompt = """Analyze the provided content across all available modalities (text, vision, audio) and provide a comprehensive analysis.

MULTI-MODAL ANALYSIS GUIDELINES:

1. **Modality Detection**: First identify which modalities are present
2. **Individual Analysis**: Analyze each modality separately
3. **Cross-Modal Integration**: Look for connections, alignments, and contradictions between modalities
4. **Unique Insights**: Identify insights that are only possible through multi-modal analysis
5. **Quality Assessment**: Evaluate the quality of each modality

ANALYSIS FOCUS AREAS:

**Text Analysis** (if present):
- Key topics and themes
- Sentiment and tone
- Named entities and important information

**Vision Analysis** (if present):
- Scene description and setting
- Objects, people, and activities
- Text visible in images
- Visual mood and atmosphere
- Colors, composition, and style

**Audio Analysis** (if present):
- Complete transcription
- Speaker identification and count
- Background sounds and audio quality
- Tone and emotional content

**Cross-Modal Integration**:
- How do the modalities complement each other?
- Are there contradictions between what's seen, heard, and read?
- What story emerges from combining all modalities?
- What would be missed if analyzing only one modality?

**People Identification**:
- Identify people across all modalities
- Match voices to faces where possible
- Note roles, relationships, and interactions

QUALITY ASSESSMENT:
- Rate each modality's quality (excellent, good, fair, poor)
- Note any processing challenges or limitations
- Provide confidence scores for your analysis"""

        if context:
            base_prompt += f"\n\nADDITIONAL CONTEXT:\n{context}"

        if media_context:
            base_prompt += f"\n\nMEDIA INFORMATION:{media_context}"

        if has_text:
            base_prompt += f"\n\nTEXT CONTENT TO ANALYZE:\n{text}"

        base_prompt += "\n\nProvide your comprehensive multi-modal analysis following the MultiModalAnalysis schema."
        
        result = {
            "prompt_text": base_prompt,
            "response_format": MultiModalAnalysis,
            "result_key": "multimodal_analysis",
        }
        
        # Add media sources
        if image_url:
            result["image_url"] = image_url
        elif image_path:
            result["image_path"] = image_path
            
        if audio_url:
            result["audio_url"] = audio_url
        elif audio_path:
            result["audio_path"] = audio_path
            
        return result

class TextOnlyPrompt(BasePromptBuilder):
    """Prompt builder for text-only analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for text analysis."""
        text = item.get("text", "")

        if not text.strip():
            raise ValueError("Text content is required for text-only analysis")
        
        analysis_prompt = f"""Analyze the following text content and provide a comprehensive analysis.

Focus on:
- Content type and classification
- Key topics and themes
- Overall sentiment
- Named entities (people, places, organizations)
- Main insights and takeaways

TEXT TO ANALYZE:
{text}

Provide your analysis following the TextOnlyAnalysis schema."""
        
        return {
            "prompt_text": analysis_prompt,
            "response_format": TextOnlyAnalysis,
            "result_key": "text_analysis",
        }

class VisionOnlyPrompt(BasePromptBuilder):
    """Prompt builder for vision-only analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for vision analysis."""
        image_path = item.get("image_path")
        image_url = item.get("image_url")
        
        if not (image_path or image_url):
            raise ValueError("Image path or URL is required for vision-only analysis")
        
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        analysis_prompt = """Analyze this image in detail and provide a comprehensive visual analysis.

Focus on:
- Detailed description of what you see
- Objects, people, and activities
- Setting and environment
- Any readable text in the image
- Dominant colors and visual style
- Overall mood and atmosphere
- Count of people visible

Provide your analysis following the VisionOnlyAnalysis schema."""
        
        result = {
            "prompt_text": analysis_prompt,
            "response_format": VisionOnlyAnalysis,
            "result_key": "vision_analysis",
        }
        
        if image_url:
            result["image_url"] = image_url
        else:
            result["image_path"] = image_path
            
        return result

class AudioOnlyPrompt(BasePromptBuilder):
    """Prompt builder for audio-only analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for audio analysis."""
        audio_path = item.get("audio_path")
        audio_url = item.get("audio_url")
        
        if not (audio_path or audio_url):
            raise ValueError("Audio path or URL is required for audio-only analysis")
        
        if audio_path and not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        analysis_prompt = """Analyze this audio content and provide a comprehensive analysis.

Focus on:
- Complete transcription of all speech
- Number of distinct speakers
- Content type and classification
- Key topics and themes discussed
- Background sounds and audio environment
- Audio quality assessment
- Overall summary and insights

Provide your analysis following the AudioOnlyAnalysis schema."""
        
        result = {
            "prompt_text": analysis_prompt,
            "response_format": AudioOnlyAnalysis,
            "result_key": "audio_analysis",
        }
        
        if audio_url:
            result["audio_url"] = audio_url
        else:
            result["audio_path"] = audio_path
            
        return result

class TextVisionPrompt(BasePromptBuilder):
    """Prompt builder for text + vision analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for text + vision analysis."""
        text = item.get("text", "")
        image_path = item.get("image_path")
        image_url = item.get("image_url")

        if not text.strip():
            raise ValueError("Text content is required for text + vision analysis")
        if not (image_path or image_url):
            raise ValueError("Image is required for text + vision analysis")
        
        analysis_prompt = f"""Analyze both the text content and the image, focusing on how they relate to each other.

Focus on:
- How the text and image complement or contradict each other
- What story emerges from combining both modalities
- Objects and elements visible in the image
- Key topics from the text
- Overall content type and purpose

TEXT CONTENT:
{text}

Provide your combined analysis following the TextVisionAnalysis schema."""
        
        result = {
            "prompt_text": analysis_prompt,
            "response_format": TextVisionAnalysis,
            "result_key": "text_vision_analysis",
        }
        
        if image_url:
            result["image_url"] = image_url
        else:
            result["image_path"] = image_path
            
        return result

class TextAudioPrompt(BasePromptBuilder):
    """Prompt builder for text + audio analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for text + audio analysis."""
        text = item.get("text", "")
        audio_path = item.get("audio_path")
        audio_url = item.get("audio_url")

        if not text.strip():
            raise ValueError("Text content is required for text + audio analysis")
        if not (audio_path or audio_url):
            raise ValueError("Audio is required for text + audio analysis")
        
        analysis_prompt = f"""Analyze both the text content and the audio, focusing on how they relate to each other.

Focus on:
- How the text and audio complement or contradict each other
- Transcription of the audio content
- Number of speakers in the audio
- Key topics from both text and audio
- Overall content type and purpose

TEXT CONTENT:
{text}

Provide your combined analysis following the TextAudioAnalysis schema."""
        
        result = {
            "prompt_text": analysis_prompt,
            "response_format": TextAudioAnalysis,
            "result_key": "text_audio_analysis",
        }
        
        if audio_url:
            result["audio_url"] = audio_url
        else:
            result["audio_path"] = audio_path
            
        return result

class VisionAudioPrompt(BasePromptBuilder):
    """Prompt builder for vision + audio analysis."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompt for vision + audio analysis."""
        image_path = item.get("image_path")
        image_url = item.get("image_url")
        audio_path = item.get("audio_path")
        audio_url = item.get("audio_url")
        
        if not (image_path or image_url):
            raise ValueError("Image is required for vision + audio analysis")
        if not (audio_path or audio_url):
            raise ValueError("Audio is required for vision + audio analysis")
        
        analysis_prompt = """Analyze both the image and audio content, focusing on how they relate to each other.

Focus on:
- How the visual and audio content complement each other
- People visible in the image and voices in the audio
- Transcription of the audio content
- Objects and setting visible in the image
- Overall content type and purpose
- Matching speakers to visible people where possible

Provide your combined analysis following the VisionAudioAnalysis schema."""
        
        result = {
            "prompt_text": analysis_prompt,
            "response_format": VisionAudioAnalysis,
            "result_key": "vision_audio_analysis",
        }
        
        if image_url:
            result["image_url"] = image_url
        else:
            result["image_path"] = image_path
            
        if audio_url:
            result["audio_url"] = audio_url
        else:
            result["audio_path"] = audio_path
            
        return result

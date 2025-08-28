from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ContentType(str, Enum):
    """Content type enumeration."""
    PRESENTATION = "presentation"
    TUTORIAL = "tutorial"
    INTERVIEW = "interview"
    PRODUCT_DEMO = "product_demo"
    NEWS_REPORT = "news_report"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    ADVERTISEMENT = "advertisement"
    DOCUMENTARY = "documentary"
    OTHER = "other"

class ModalityQuality(str, Enum):
    """Quality assessment for each modality."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class PersonInfo(BaseModel):
    """Information about a person detected in the content."""
    name: Optional[str] = Field(default=None, description="Person's name if mentioned or recognizable")
    role: Optional[str] = Field(default=None, description="Person's role or title")
    description: str = Field(description="Physical description or characteristics")
    speaking: bool = Field(description="Whether this person is speaking in the audio")
    visible: bool = Field(description="Whether this person is visible in the image/video")

class MultiModalAnalysis(BaseModel):
    """Comprehensive analysis combining text, vision, and audio modalities."""
    
    # Overall content analysis
    content_type: ContentType = Field(description="Type of content being analyzed")
    title: Optional[str] = Field(default=None, description="Title or subject if determinable")
    summary: str = Field(description="Comprehensive summary combining all modalities")
    
    # Modality detection and quality
    modalities_present: List[str] = Field(description="List of modalities detected (text, vision, audio)")
    text_quality: Optional[ModalityQuality] = Field(default=None, description="Quality of text content")
    vision_quality: Optional[ModalityQuality] = Field(default=None, description="Quality of visual content")
    audio_quality: Optional[ModalityQuality] = Field(default=None, description="Quality of audio content")
    
    # Text analysis
    text_content: Optional[str] = Field(default=None, description="Extracted or provided text content")
    key_topics: List[str] = Field(default_factory=list, description="Main topics from text analysis")
    
    # Vision analysis
    visual_description: Optional[str] = Field(default=None, description="Description of visual content")
    objects_detected: List[str] = Field(default_factory=list, description="Objects or items visible in image")
    scene_setting: Optional[str] = Field(default=None, description="Setting or environment description")
    text_in_image: Optional[str] = Field(default=None, description="Any text visible in the image")
    
    # Audio analysis
    audio_transcript: Optional[str] = Field(default=None, description="Transcript of spoken content")
    speaker_count: Optional[int] = Field(default=None, description="Number of distinct speakers")
    background_sounds: List[str] = Field(default_factory=list, description="Non-speech sounds detected")
    
    # People analysis
    people_detected: List[PersonInfo] = Field(default_factory=list, description="People identified across modalities")
    
    # Cross-modal insights
    modality_alignment: str = Field(description="How well the different modalities align or complement each other")
    contradictions: List[str] = Field(default_factory=list, description="Any contradictions between modalities")
    unique_insights: List[str] = Field(default_factory=list, description="Insights only possible through multi-modal analysis")
    
    # Metadata
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in the analysis")
    processing_notes: Optional[str] = Field(default=None, description="Notes about processing challenges or limitations")

class TextOnlyAnalysis(BaseModel):
    """Analysis for text-only content."""
    content_type: ContentType = Field(description="Type of text content")
    summary: str = Field(description="Summary of the text")
    key_topics: List[str] = Field(description="Main topics discussed")
    sentiment: str = Field(description="Overall sentiment")
    entities: List[str] = Field(default_factory=list, description="Named entities mentioned")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

class VisionOnlyAnalysis(BaseModel):
    """Analysis for vision-only content."""
    visual_description: str = Field(description="Detailed description of the image")
    objects_detected: List[str] = Field(description="Objects or items visible")
    scene_setting: str = Field(description="Setting or environment")
    people_count: int = Field(description="Number of people visible")
    text_in_image: Optional[str] = Field(default=None, description="Any readable text")
    colors_dominant: List[str] = Field(default_factory=list, description="Dominant colors")
    mood_atmosphere: str = Field(description="Overall mood or atmosphere")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

class AudioOnlyAnalysis(BaseModel):
    """Analysis for audio-only content."""
    transcript: str = Field(description="Full transcript of audio")
    speaker_count: int = Field(description="Number of distinct speakers")
    content_type: ContentType = Field(description="Type of audio content")
    summary: str = Field(description="Summary of audio content")
    key_topics: List[str] = Field(description="Main topics discussed")
    background_sounds: List[str] = Field(default_factory=list, description="Non-speech sounds")
    audio_quality: ModalityQuality = Field(description="Quality of audio")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

class TextVisionAnalysis(BaseModel):
    """Analysis combining text and vision modalities."""
    content_type: ContentType = Field(description="Type of content")
    summary: str = Field(description="Combined summary from text and vision")
    text_content: str = Field(description="Text content analyzed")
    visual_description: str = Field(description="Visual content description")
    text_image_alignment: str = Field(description="How text and image relate to each other")
    objects_detected: List[str] = Field(description="Objects visible in image")
    key_topics: List[str] = Field(description="Topics from text analysis")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

class TextAudioAnalysis(BaseModel):
    """Analysis combining text and audio modalities."""
    content_type: ContentType = Field(description="Type of content")
    summary: str = Field(description="Combined summary from text and audio")
    text_content: str = Field(description="Text content analyzed")
    audio_transcript: str = Field(description="Audio transcript")
    text_audio_alignment: str = Field(description="How text and audio relate to each other")
    speaker_count: int = Field(description="Number of speakers in audio")
    key_topics: List[str] = Field(description="Topics from combined analysis")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

class VisionAudioAnalysis(BaseModel):
    """Analysis combining vision and audio modalities."""
    content_type: ContentType = Field(description="Type of content")
    summary: str = Field(description="Combined summary from vision and audio")
    visual_description: str = Field(description="Visual content description")
    audio_transcript: str = Field(description="Audio transcript")
    vision_audio_alignment: str = Field(description="How visual and audio content relate")
    people_detected: List[PersonInfo] = Field(description="People identified across both modalities")
    objects_detected: List[str] = Field(description="Objects visible in image")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class AudioType(str, Enum):
    """Audio content type enumeration."""
    SPEECH = "speech"
    MUSIC = "music"
    PODCAST = "podcast"
    INTERVIEW = "interview"
    LECTURE = "lecture"
    MEETING = "meeting"
    PHONE_CALL = "phone_call"
    AUDIOBOOK = "audiobook"
    NEWS = "news"
    OTHER = "other"

class AudioQuality(str, Enum):
    """Audio quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class SpeakerInfo(BaseModel):
    """Information about a speaker in the audio."""
    speaker_id: str = Field(description="Unique identifier for the speaker (e.g., 'Speaker 1', 'Host', 'Guest')")
    gender: Optional[str] = Field(default=None, description="Perceived gender of the speaker")
    estimated_age_range: Optional[str] = Field(default=None, description="Estimated age range (e.g., '20-30', '40-50')")
    accent_or_language: Optional[str] = Field(default=None, description="Detected accent or language")
    speaking_style: Optional[str] = Field(default=None, description="Speaking style (formal, casual, energetic, etc.)")

class AudioAnalysis(BaseModel):
    """Comprehensive audio content analysis."""
    
    # Basic information
    audio_type: AudioType = Field(description="Type of audio content")
    duration_estimate: Optional[str] = Field(default=None, description="Estimated duration if detectable")
    language: str = Field(description="Primary language detected")
    
    # Content analysis
    title: Optional[str] = Field(default=None, description="Title or subject if mentioned")
    summary: str = Field(description="Brief summary of the audio content")
    transcript: str = Field(description="Full transcript of the audio")
    
    # Key information extraction
    main_topics: List[str] = Field(default_factory=list, description="Main topics discussed")
    key_points: List[str] = Field(default_factory=list, description="Key points or takeaways")
    mentioned_names: List[str] = Field(default_factory=list, description="Names of people mentioned")
    mentioned_places: List[str] = Field(default_factory=list, description="Places or locations mentioned")
    mentioned_organizations: List[str] = Field(default_factory=list, description="Organizations or companies mentioned")
    
    # Speaker analysis
    speakers: List[SpeakerInfo] = Field(default_factory=list, description="Information about speakers")
    speaker_count: int = Field(description="Number of distinct speakers detected")
    
    # Technical quality
    audio_quality: AudioQuality = Field(description="Overall audio quality assessment")
    background_noise: bool = Field(description="Whether significant background noise is present")
    clarity_issues: List[str] = Field(default_factory=list, description="Any clarity or audio issues noted")
    
    # Sentiment and tone
    overall_tone: str = Field(description="Overall tone (professional, casual, emotional, etc.)")
    sentiment: str = Field(description="Overall sentiment (positive, negative, neutral)")
    
    # Confidence and metadata
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis accuracy (0-1)")
    processing_notes: Optional[str] = Field(default=None, description="Any notes about processing challenges")

class AudioEvaluation(BaseModel):
    """Evaluation of audio extraction quality."""
    
    # Accuracy metrics
    transcript_accuracy: bool = Field(description="Is the transcript accurate?")
    speaker_identification: bool = Field(description="Are speakers correctly identified?")
    content_understanding: bool = Field(description="Is the content properly understood?")
    
    # Completeness metrics
    complete_transcript: bool = Field(description="Is the transcript complete?")
    all_speakers_identified: bool = Field(description="Are all speakers identified?")
    key_info_extracted: bool = Field(description="Is key information properly extracted?")
    
    # Quality metrics
    schema_compliance: bool = Field(description="Does output follow schema correctly?")
    reasonable_confidence: bool = Field(description="Is confidence score reasonable?")
    appropriate_categorization: bool = Field(description="Is audio type correctly categorized?")
    
    # Overall assessment
    overall_quality: str = Field(description="Overall quality: excellent, good, fair, or poor")
    missing_elements: List[str] = Field(default_factory=list, description="Elements that should have been extracted")
    incorrect_elements: List[str] = Field(default_factory=list, description="Incorrectly extracted elements")
    
    # Feedback
    reasoning: Optional[str] = Field(default=None, description="Explanation of the evaluation")
    improvement_suggestions: Optional[str] = Field(default=None, description="Suggestions for better extraction")

class AudioSummary(BaseModel):
    """Simplified audio summary for quick processing."""
    summary: str = Field(description="Brief summary of audio content")
    transcript: str = Field(description="Full transcript")
    main_topic: str = Field(description="Primary topic or subject")
    speaker_count: int = Field(description="Number of speakers")
    audio_type: AudioType = Field(description="Type of audio content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

class MeetingAnalysis(BaseModel):
    """Specialized analysis for meeting audio."""
    meeting_type: str = Field(description="Type of meeting (standup, planning, review, etc.)")
    participants: List[str] = Field(description="List of participant names if mentioned")
    agenda_items: List[str] = Field(default_factory=list, description="Agenda items discussed")
    action_items: List[str] = Field(default_factory=list, description="Action items identified")
    decisions_made: List[str] = Field(default_factory=list, description="Decisions made during meeting")
    next_steps: List[str] = Field(default_factory=list, description="Next steps or follow-ups")
    meeting_summary: str = Field(description="Overall meeting summary")
    transcript: str = Field(description="Full meeting transcript")

class PodcastAnalysis(BaseModel):
    """Specialized analysis for podcast audio."""
    podcast_name: Optional[str] = Field(default=None, description="Podcast name if mentioned")
    episode_title: Optional[str] = Field(default=None, description="Episode title if mentioned")
    host_names: List[str] = Field(default_factory=list, description="Host names")
    guest_names: List[str] = Field(default_factory=list, description="Guest names")
    episode_summary: str = Field(description="Episode summary")
    key_quotes: List[str] = Field(default_factory=list, description="Notable quotes from the episode")
    topics_covered: List[str] = Field(description="Topics covered in the episode")
    transcript: str = Field(description="Full episode transcript")
    sponsor_mentions: List[str] = Field(default_factory=list, description="Sponsors or ads mentioned")

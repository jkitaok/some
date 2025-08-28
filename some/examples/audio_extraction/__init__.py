"""
Audio Extraction Example

This package demonstrates how to use the some library for extracting structured
information from audio files using OpenAI's audio-capable models and the instructor library.

Key components:
- AudioAnalysis: Comprehensive schema for audio content analysis
- AudioSummary: Simplified schema for quick audio processing
- MeetingAnalysis: Specialized schema for meeting recordings
- PodcastAnalysis: Specialized schema for podcast episodes
- AudioAnalysisPrompt: Prompt builder for comprehensive audio analysis
- AudioSummaryPrompt: Prompt builder for quick audio summarization
- run_audio_extraction: Main execution script

Example usage:
    from some.examples.audio_extraction import main
    results = main()
"""

from .audio_schema import (
    AudioAnalysis, AudioEvaluation, AudioSummary, AudioType, AudioQuality,
    SpeakerInfo, MeetingAnalysis, PodcastAnalysis
)
from .audio_prompt import (
    AudioAnalysisPrompt, AudioSummaryPrompt, AudioEvaluationPrompt,
    MeetingAnalysisPrompt, PodcastAnalysisPrompt
)
from .run_audio_extraction import main, load_sample_data, format_evaluation_record

__all__ = [
    "AudioAnalysis",
    "AudioEvaluation", 
    "AudioSummary",
    "AudioType",
    "AudioQuality",
    "SpeakerInfo",
    "MeetingAnalysis",
    "PodcastAnalysis",
    "AudioAnalysisPrompt",
    "AudioSummaryPrompt",
    "AudioEvaluationPrompt",
    "MeetingAnalysisPrompt",
    "PodcastAnalysisPrompt",
    "main",
    "load_sample_data",
    "format_evaluation_record"
]

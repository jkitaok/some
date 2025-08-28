"""
Multi-Modal Extraction Example

This package demonstrates how to use the some library for extracting structured
information from content that combines text, vision, and audio modalities.

Key components:
- MultiModalAnalysis: Comprehensive schema for multi-modal content analysis
- Individual modality schemas: TextOnlyAnalysis, VisionOnlyAnalysis, AudioOnlyAnalysis
- Dual modality schemas: TextVisionAnalysis, TextAudioAnalysis, VisionAudioAnalysis
- Prompt builders for all modality combinations
- Comprehensive testing across all modality combinations

Example usage:
    from some.examples.multimodal_extraction import main
    results = main()
    
    # Or test specific combinations
    from some.examples.multimodal_extraction import TextVisionPrompt
    prompt_builder = TextVisionPrompt()
"""

from .multimodal_schema import (
    MultiModalAnalysis, TextOnlyAnalysis, VisionOnlyAnalysis, AudioOnlyAnalysis,
    TextVisionAnalysis, TextAudioAnalysis, VisionAudioAnalysis,
    ContentType, ModalityQuality, PersonInfo
)
from .multimodal_prompt import (
    MultiModalPrompt, TextOnlyPrompt, VisionOnlyPrompt, AudioOnlyPrompt,
    TextVisionPrompt, TextAudioPrompt, VisionAudioPrompt
)
from .run_multimodal_extraction import main, load_sample_data, determine_prompt_builder

__all__ = [
    # Schemas
    "MultiModalAnalysis",
    "TextOnlyAnalysis",
    "VisionOnlyAnalysis", 
    "AudioOnlyAnalysis",
    "TextVisionAnalysis",
    "TextAudioAnalysis",
    "VisionAudioAnalysis",
    "ContentType",
    "ModalityQuality",
    "PersonInfo",
    
    # Prompt builders
    "MultiModalPrompt",
    "TextOnlyPrompt",
    "VisionOnlyPrompt",
    "AudioOnlyPrompt", 
    "TextVisionPrompt",
    "TextAudioPrompt",
    "VisionAudioPrompt",
    
    # Main functions
    "main",
    "load_sample_data",
    "determine_prompt_builder"
]

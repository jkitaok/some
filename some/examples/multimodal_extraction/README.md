# Multi-Modal Extraction Example

This example demonstrates comprehensive multi-modal content analysis using the `some` library. It showcases how to extract structured information from content that combines **text**, **vision**, and **audio** modalities in all possible combinations.

## 🎯 What This Example Does

This is the most comprehensive example in the `some` library, demonstrating:

1. **All Modality Combinations**: 
   - Single: Text-only, Vision-only, Audio-only
   - Dual: Text+Vision, Text+Audio, Vision+Audio  
   - Triple: Text+Vision+Audio

2. **Cross-Modal Analysis**: 
   - Alignment detection between modalities
   - Contradiction identification
   - Unique insights from multi-modal combination

3. **Intelligent Processing**:
   - Automatic modality detection
   - Appropriate schema selection
   - Quality assessment per modality

## 📁 Project Structure

```
multimodal_extraction/
├── __init__.py                     # Package initialization
├── README.md                       # This documentation
├── multimodal_schema.py            # Comprehensive schemas for all combinations
├── multimodal_prompt.py            # Prompt builders for all combinations
├── run_multimodal_extraction.py    # Main execution script
├── test_multimodal.py              # Comprehensive test suite
└── input_dataset/                  # Sample data for all combinations
    ├── README.md                   # Dataset documentation
    ├── sample_multimodal.json      # Sample data with all combinations
    ├── images/                     # Image files for vision analysis
    │   ├── tech_presentation.jpg   # Technology presentation
    │   ├── product_demo.jpg        # Product demonstration
    │   ├── interview_setup.jpg     # Interview setup
    │   ├── ml_tutorial.jpg         # ML tutorial slide
    │   ├── solar_panels.jpg        # Solar technology
    │   ├── smart_home_device.jpg   # Smart home product
    │   └── rainforest.jpg          # Nature documentary
    └── audio_files/                # Local audio files (optional)
        ├── presentation_audio.wav  # Presentation narration
        ├── interview_audio.wav     # Interview recording
        └── tutorial_audio.wav      # Tutorial explanation
```

## 🚀 Quick Start

### Prerequisites

1. Install required libraries:
   ```bash
   pip install instructor
   ```

2. Set up API access:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

### Running the Example

1. **Comprehensive test**:
   ```bash
   cd some/examples/multimodal_extraction
   python test_multimodal.py
   ```

2. **Full multi-modal analysis**:
   ```bash
   python run_multimodal_extraction.py
   ```

3. **As a Python module**:
   ```python
   from some.examples.multimodal_extraction import main
   results = main()
   ```

## 📊 Schema Architecture

### Core Multi-Modal Schema
- **MultiModalAnalysis**: Complete analysis across all modalities
- **PersonInfo**: Cross-modal person identification
- **ContentType**: Unified content classification

### Single Modality Schemas
- **TextOnlyAnalysis**: Pure text analysis
- **VisionOnlyAnalysis**: Pure image analysis  
- **AudioOnlyAnalysis**: Pure audio analysis

### Dual Modality Schemas
- **TextVisionAnalysis**: Text + image combination
- **TextAudioAnalysis**: Text + audio combination
- **VisionAudioAnalysis**: Image + audio combination

## 🎭 Modality Combinations

### 1. Single Modality
```python
# Text only
{"prompt_text": "AI is transforming industries..."}

# Vision only  
{"image_path": "presentation.jpg"}

# Audio only
{"audio_url": "speech.wav"}
```

### 2. Dual Modality
```python
# Text + Vision
{
    "prompt_text": "Product presentation slides...",
    "image_path": "slide.jpg"
}

# Text + Audio
{
    "prompt_text": "Podcast description...", 
    "audio_url": "episode.wav"
}

# Vision + Audio
{
    "image_path": "interview.jpg",
    "audio_url": "conversation.wav"
}
```

### 3. Triple Modality
```python
# Text + Vision + Audio (Complete multimedia)
{
    "prompt_text": "Tutorial description...",
    "image_path": "tutorial_slide.jpg", 
    "audio_url": "narration.wav"
}
```

## 🔧 Advanced Features

### Automatic Modality Detection
The system automatically detects available modalities:

```python
from some.examples.multimodal_extraction import determine_prompt_builder

# Automatically selects appropriate prompt builder
modalities = ["text", "vision", "audio"]
prompt_builder = determine_prompt_builder(modalities)
```

### Cross-Modal Insights
Unique analysis only possible through multi-modal combination:

- **Alignment Detection**: How well modalities complement each other
- **Contradiction Identification**: Conflicts between modalities
- **Enhanced Context**: Information enriched by multiple sources
- **Person Matching**: Matching voices to faces across modalities

### Quality Assessment
Per-modality quality evaluation:

- **Excellent**: Crystal clear, perfect for analysis
- **Good**: High quality with minor issues
- **Fair**: Usable but with some limitations
- **Poor**: Significant quality issues affecting analysis

## 🎯 Use Cases

### Educational Content
- **Online Courses**: Video lectures with slides and transcripts
- **Tutorials**: Step-by-step guides with visual and audio components
- **Documentaries**: Narrated content with rich visuals

### Business Applications
- **Presentations**: Slide decks with speaker notes and recordings
- **Product Demos**: Marketing materials with multiple media types
- **Meeting Analysis**: Video conferences with slides and audio

### Media Analysis
- **News Reports**: Articles with images and video/audio content
- **Interviews**: Video interviews with transcripts and visual context
- **Advertisements**: Multi-media marketing campaigns

### Research Applications
- **Content Analysis**: Academic research on multimedia content
- **Accessibility**: Creating comprehensive descriptions for accessibility
- **Archive Processing**: Digitizing and analyzing historical multimedia

## 📈 Performance Optimization

### Efficient Processing
- **Batch Processing**: Groups items by modality combination
- **Smart Routing**: Uses appropriate models for each combination
- **Parallel Execution**: Concurrent processing where possible

### Cost Management
- **Model Selection**: Chooses cost-effective models when possible
- **Token Optimization**: Efficient prompt construction
- **Quality Scaling**: Adjusts processing based on content quality

## 🛠️ Customization

### Adding New Modalities
The architecture is extensible for future modalities:

```python
# Example: Adding video modality
class VideoAnalysis(BaseModel):
    video_description: str
    scene_changes: List[str]
    motion_analysis: str
```

### Custom Content Types
Extend the ContentType enum for domain-specific content:

```python
class ContentType(str, Enum):
    # Existing types...
    MEDICAL_CONSULTATION = "medical_consultation"
    LEGAL_PROCEEDING = "legal_proceeding"
    SCIENTIFIC_PRESENTATION = "scientific_presentation"
```

### Domain-Specific Schemas
Create specialized schemas for specific domains:

```python
class MedicalMultiModalAnalysis(MultiModalAnalysis):
    medical_terminology: List[str]
    diagnosis_mentioned: Optional[str]
    treatment_discussed: bool
```

## 🧪 Testing

The example includes comprehensive testing:

```bash
# Run all tests
python test_multimodal.py

# Test specific combinations
python -c "from test_multimodal import test_text_vision_extraction; test_text_vision_extraction()"
```

### Test Coverage
- ✅ Schema validation for all combinations
- ✅ Modality detection accuracy
- ✅ Single modality processing
- ✅ Dual modality combinations
- ✅ Triple modality integration
- ✅ Cross-modal analysis quality

## 🔄 Integration

Easy integration into larger systems:

```python
from some.examples.multimodal_extraction import MultiModalPrompt
from some.inference import get_language_model

def analyze_multimedia_content(text=None, image=None, audio=None):
    # Build input
    content = {}
    if text: content["prompt_text"] = text
    if image: content["image_path"] = image  
    if audio: content["audio_path"] = audio
    
    # Process
    prompt_builder = MultiModalPrompt()
    extraction_input = prompt_builder.build(content)
    
    lm = get_language_model(provider="instructor", model="gpt-4o")
    results, _, _ = lm.generate([extraction_input])
    
    return results[0].get("multimodal_analysis", {})
```

## 📚 Further Reading

- [Vision Extraction Example](../vision_extraction/README.md) - Image-focused analysis
- [Audio Extraction Example](../audio_extraction/README.md) - Audio-focused analysis
- [Instructor Documentation](https://python.useinstructor.com/) - Multi-modal capabilities
- [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) - Vision API details

## 🎉 Key Achievements

This example demonstrates the most advanced multi-modal AI capabilities:

✅ **Complete Coverage**: All possible modality combinations  
✅ **Intelligent Routing**: Automatic selection of appropriate processing  
✅ **Cross-Modal Analysis**: Insights impossible with single modalities  
✅ **Production Ready**: Robust error handling and quality assessment  
✅ **Extensible Architecture**: Easy to add new modalities and domains  
✅ **Comprehensive Testing**: Full test coverage for all combinations  

This represents the cutting edge of multi-modal AI content analysis, providing a foundation for the most sophisticated multimedia understanding applications.

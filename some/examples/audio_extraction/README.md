# Audio Extraction Example

This example demonstrates how to use the `some` library for extracting structured information from audio files using OpenAI's audio-capable models and the instructor library. It showcases audio AI capabilities for transcription, content analysis, and information extraction from various audio types.

## 🎯 What This Example Does

1. **Audio Content Analysis**: Analyzes audio files to extract comprehensive information including:
   - Complete transcription of spoken content
   - Speaker identification and analysis
   - Key topics, names, places, and organizations mentioned
   - Audio quality assessment and technical details
   - Sentiment and tone analysis

2. **Specialized Audio Processing**: Supports different audio types with tailored analysis:
   - **Meetings**: Action items, decisions, participants
   - **Podcasts**: Hosts, guests, episode information, key quotes
   - **Interviews**: Structured Q&A analysis
   - **Lectures**: Educational content extraction
   - **Phone Calls**: Customer service analysis

3. **Quality Evaluation**: Automatically evaluates extraction accuracy by:
   - Assessing transcript completeness and accuracy
   - Validating speaker identification
   - Checking content understanding quality

## 📁 Project Structure

```
audio_extraction/
├── __init__.py                     # Package initialization
├── README.md                       # This documentation
├── audio_schema.py                 # Pydantic schemas for audio data
├── audio_prompt.py                 # Prompt builders for extraction tasks
├── run_audio_extraction.py         # Main execution script
├── test_audio_extraction.py        # Simple test script
└── input_dataset/                  # Sample data and audio files
    ├── README.md                   # Dataset documentation
    ├── sample_audio.json           # Sample audio metadata
    └── audio_files/                # Local audio file storage
        ├── team_meeting.wav        # Meeting example
        ├── podcast_episode.wav     # Podcast example
        ├── customer_call.wav       # Phone call example
        ├── lecture_excerpt.wav     # Educational content
        └── interview.wav           # Interview example
```

## 🚀 Quick Start

### Prerequisites

1. Install the `instructor` library for audio support:
   ```bash
   pip install instructor
   ```

2. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. Ensure access to OpenAI's audio models (gpt-4o-audio-preview)

### Running the Example

1. **Simple test with remote audio**:
   ```bash
   cd some/examples/audio_extraction
   python test_audio_extraction.py
   ```

2. **Full example with sample data**:
   ```bash
   python run_audio_extraction.py
   ```

3. **As a Python module**:
   ```python
   from some.examples.audio_extraction import main
   results = main()
   ```

### Custom Usage

```python
from some.examples.audio_extraction import AudioAnalysisPrompt, AudioAnalysis
from some.inference import get_language_model

# Analyze audio from URL
audio_item = {
    "audio_url": "https://example.com/audio.wav",
    "context": "Meeting recording",
    "expected_type": "meeting"
}

# Build extraction input
prompt_builder = AudioAnalysisPrompt()
extraction_input = prompt_builder.build(audio_item)

# Run extraction
lm = get_language_model(provider="audio_instructor", model="gpt-4o-audio-preview")
results, workers, timing = lm.generate([extraction_input])
```

## 📊 Schema Overview

### AudioAnalysis
Comprehensive audio analysis schema with fields for:
- **Basic Info**: audio_type, duration_estimate, language
- **Content**: title, summary, transcript, main_topics, key_points
- **Entities**: mentioned_names, mentioned_places, mentioned_organizations
- **Speakers**: speaker information, count, characteristics
- **Quality**: audio_quality, background_noise, clarity_issues
- **Sentiment**: overall_tone, sentiment, confidence_score

### Specialized Schemas
- **MeetingAnalysis**: Meeting-specific fields (action_items, decisions, participants)
- **PodcastAnalysis**: Podcast-specific fields (hosts, guests, episode_info, sponsors)
- **AudioSummary**: Simplified schema for quick processing

## 🔧 Audio Model Support

### Instructor Integration
Uses the instructor library with OpenAI's audio models:

```python
from instructor.multimodal import Audio

# From URL
Audio.from_url("https://example.com/audio.wav")

# From local file
Audio.from_path("/path/to/audio.wav")
```

### OpenAI Direct Integration
Direct OpenAI API usage with audio modalities:

```python
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[{"role": "user", "content": "Analyze this audio..."}]
)
```

## 🎵 Supported Audio Formats

- **WAV** (.wav) - Recommended for best compatibility
- **MP3** (.mp3) - Widely supported
- **M4A** (.m4a) - Apple audio format
- **FLAC** (.flac) - Lossless compression

## 📈 Performance Optimization

### Processing Guidelines
- **File Size**: Keep under OpenAI's API limits
- **Duration**: 1-30 minutes optimal for testing
- **Quality**: Clear speech improves accuracy
- **Format**: WAV recommended for best results

### Cost Management
- Uses efficient audio processing models
- Tracks token usage and costs
- Provides detailed cost breakdowns
- Supports batch processing for multiple files

## 🛠️ Troubleshooting

### Common Issues

1. **"instructor library not found"**
   ```bash
   pip install instructor
   ```

2. **"Audio model not available"**
   - Ensure API key has access to gpt-4o-audio-preview
   - Check OpenAI account status and billing

3. **"Audio file not found"**
   - Verify file paths in sample_audio.json
   - Ensure audio files exist in input_dataset/audio_files/

4. **"Remote audio URL failed"**
   - Check internet connection
   - Verify URL accessibility
   - Try with local files instead

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Use Cases

This example is ideal for:

- **Meeting Transcription**: Automated meeting notes and action items
- **Podcast Analysis**: Content extraction and episode summaries
- **Customer Service**: Call analysis and quality assessment
- **Educational Content**: Lecture transcription and key point extraction
- **Interview Processing**: Structured interview analysis
- **Content Creation**: Audio content repurposing and summarization

## 🔄 Integration

The audio extraction components can be easily integrated into larger systems:

```python
# Custom audio processing pipeline
from some.examples.audio_extraction import AudioAnalysisPrompt

def process_audio_file(audio_path: str) -> dict:
    prompt_builder = AudioAnalysisPrompt()
    input_data = prompt_builder.build({"audio_path": audio_path})
    
    # Use your preferred language model
    lm = get_language_model(provider="audio_instructor")
    results, _, _ = lm.generate([input_data])
    
    return results[0].get("audio_analysis", {})
```

## 📚 Further Reading

- [OpenAI Audio API Documentation](https://platform.openai.com/docs/guides/audio)
- [Instructor Library Documentation](https://python.useinstructor.com/)
- [Pydantic Schema Documentation](https://docs.pydantic.dev/)
- [Some Library Documentation](../../README.md)

## 🧪 Testing

The example includes a test that works out of the box using the Gettysburg Address audio file from the instructor library repository. This allows you to test the audio extraction functionality without needing to provide your own audio files.

Run the test:
```bash
python test_audio_extraction.py
```

This will test both schema validation and actual audio processing using the remote audio file.

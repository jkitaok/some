# Multi-Modal Dataset

This directory contains sample data for testing comprehensive multi-modal extraction capabilities across text, vision, and audio modalities.

## Directory Structure

```
input_dataset/
├── images/                        # Image files for vision analysis
│   ├── tech_presentation.jpg      # Technology presentation slide
│   ├── product_demo.jpg           # Product demonstration screenshot
│   ├── interview_setup.jpg        # Interview recording setup
│   ├── ml_tutorial.jpg            # Machine learning tutorial slide
│   ├── solar_panels.jpg           # Solar panel technology image
│   ├── smart_home_device.jpg      # Smart home product image
│   └── rainforest.jpg             # Amazon rainforest documentary image
├── audio_files/                   # Local audio files (optional)
│   ├── presentation_audio.wav     # Presentation narration
│   ├── interview_audio.wav        # Interview recording
│   └── tutorial_audio.wav         # Tutorial explanation
├── sample_multimodal.json         # Sample data with all combinations
└── README.md                      # This file
```

## Sample Data Format

The `sample_multimodal.json` file contains test cases for all modality combinations:

```json
{
  "id": "unique_sample_id",
  "prompt_text": "Text content to analyze (optional)",
  "image_path": "path/to/image.jpg (optional)",
  "image_url": "https://example.com/image.jpg (optional)",
  "audio_path": "path/to/audio.wav (optional)",
  "audio_url": "https://example.com/audio.wav (optional)",
  "modalities": ["text", "vision", "audio"],
  "context": "Additional context about the content",
  "expected_analysis": {
    "content_type": "expected_type",
    "key_topics": ["topic1", "topic2"]
  }
}
```

## Modality Combinations

The dataset includes examples for all possible combinations:

### Single Modality
- **Text Only**: Pure text analysis (articles, descriptions, etc.)
- **Vision Only**: Image analysis without accompanying text or audio
- **Audio Only**: Audio content analysis (speeches, recordings, etc.)

### Dual Modality
- **Text + Vision**: Articles with images, presentations with slides
- **Text + Audio**: Podcast descriptions with audio, transcripts with recordings
- **Vision + Audio**: Video content, interviews with visual setup

### Triple Modality
- **Text + Vision + Audio**: Complete multimedia content (videos with descriptions, presentations with slides and narration, documentaries, etc.)

## Content Types Covered

- **Educational**: Tutorials, lectures, explanatory content
- **Presentation**: Business presentations, product demos
- **Interview**: Q&A sessions, conversations
- **News Report**: Breaking news, journalism
- **Advertisement**: Product marketing, promotional content
- **Documentary**: Educational documentaries, nature content
- **Entertainment**: Creative content, media

## Remote Audio Source

For immediate testing without local audio files, the examples use:
- **Gettysburg Address**: `https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav`

This allows testing audio functionality without requiring local audio files.

## Adding Your Own Content

### Images
Place image files in the `images/` directory:
- **Formats**: JPG, PNG, WebP, GIF
- **Size**: Recommended 1024x1024 or smaller
- **Content**: Clear, relevant to your use case

### Audio Files
Place audio files in the `audio_files/` directory:
- **Formats**: WAV (recommended), MP3, M4A
- **Quality**: Clear speech, minimal background noise
- **Length**: 1-30 minutes for optimal processing

### Sample Data
Update `sample_multimodal.json` with your content:
1. Add new entries with unique IDs
2. Specify which modalities are present
3. Provide context and expected analysis
4. Use absolute or relative paths consistently

## Multi-Modal Analysis Benefits

Combining modalities provides unique insights:

1. **Cross-Modal Validation**: Verify information across different sources
2. **Enhanced Understanding**: Get complete picture from partial information
3. **Context Enrichment**: Audio tone + visual cues + text content
4. **Contradiction Detection**: Identify misalignments between modalities
5. **Comprehensive Extraction**: Extract information only available through combination

## Testing Guidelines

For best results:
- **Complementary Content**: Use content where modalities support each other
- **Clear Quality**: Ensure good quality for each modality
- **Relevant Context**: Provide meaningful context for better analysis
- **Realistic Scenarios**: Use real-world content combinations

## API Requirements

Multi-modal analysis requires:
- **OpenAI API Key**: For models with vision/audio capabilities
- **Instructor Library**: For structured multi-modal output
- **Model Access**: GPT-4o or similar multi-modal models

## Performance Notes

- **Processing Time**: Multi-modal analysis takes longer than single modality
- **Token Usage**: Higher token consumption due to multiple inputs
- **Quality Impact**: Poor quality in one modality affects overall analysis
- **Model Limitations**: Not all models support all modality combinations

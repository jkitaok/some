# Audio Files Dataset

This directory contains sample audio files for testing audio extraction capabilities using OpenAI's audio models and the instructor library.

## Directory Structure

```
input_dataset/
├── audio_files/                   # Audio file storage
│   ├── team_meeting.wav          # Meeting example
│   ├── podcast_episode.wav       # Podcast example
│   ├── customer_call.wav         # Phone call example
│   ├── lecture_excerpt.wav       # Educational content
│   └── interview.wav             # Interview example
├── sample_audio.json             # Sample data with expected results
└── README.md                     # This file
```

## Sample Data Format

The `sample_audio.json` file contains test cases with the following structure:

```json
{
  "id": "unique_audio_id",
  "audio_path": "relative/path/to/audio.wav",
  "audio_url": "https://example.com/audio.wav",
  "expected_type": "meeting|podcast|speech|interview|lecture",
  "context": "Additional context about the audio",
  "expected_content": {
    "audio_type": "expected_type",
    "speaker_count": 2,
    "main_topics": ["topic1", "topic2"]
  }
}
```

## Audio Sources

### Remote Audio
- **Gettysburg Address**: Uses the sample audio from the instructor library repository
- **URL**: `https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav`

### Local Audio Files
The following audio files should be placed in the `audio_files/` directory:

- `team_meeting.wav` - Team standup meeting recording
- `podcast_episode.wav` - Technology podcast episode
- `customer_call.wav` - Customer service call
- `lecture_excerpt.wav` - University lecture excerpt
- `interview.wav` - Job interview recording

## Adding Your Own Audio

To test with your own audio files:

1. Add audio files to the `audio_files/` directory
2. Update `sample_audio.json` with corresponding entries
3. Supported formats: WAV, MP3, M4A, FLAC
4. Recommended: WAV format for best compatibility
5. Maximum file size: Check OpenAI API limits

## Audio Guidelines

For best extraction results, use audio that:

- **Clear Speech**: Speakers should be clearly audible
- **Good Quality**: Minimal background noise and distortion
- **Appropriate Length**: 1-30 minutes for testing (API limits apply)
- **Single Language**: Primarily one language per file
- **Structured Content**: Meetings, interviews, lectures work well

## Expected Audio Types

The extraction system recognizes these audio types:

- `speech` - Single speaker presentations, speeches
- `meeting` - Multi-speaker business meetings
- `podcast` - Podcast episodes with hosts and guests
- `interview` - Interview recordings
- `lecture` - Educational content, presentations
- `phone_call` - Phone conversations
- `audiobook` - Narrated book content
- `news` - News broadcasts or reports
- `other` - Miscellaneous audio content

## API Requirements

This example requires:

1. **OpenAI API Key**: For audio processing with GPT-4o-audio-preview
2. **Instructor Library**: For structured output from audio
3. **Audio Model Access**: Ensure your API key has access to audio models

Set your API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Processing Notes

- Audio files are processed using OpenAI's audio-capable models
- The system can handle both local files and remote URLs
- Processing time depends on audio length and complexity
- Costs apply based on OpenAI's audio processing pricing
- Large files may need to be chunked for processing

## Testing with Sample Data

The example includes one remote audio file (Gettysburg Address) that works out of the box for testing the audio extraction functionality without requiring local audio files.

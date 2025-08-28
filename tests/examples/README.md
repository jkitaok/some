# Example Tests

This directory contains comprehensive tests for all extraction examples in the `some` library.

## 📁 Directory Structure

```
tests/examples/
├── README.md                           # This documentation
├── __init__.py                         # Package initialization
├── run_example_tests.py                # Test runner for all example tests
├── test_dataset_standardization.py     # Dataset format validation tests
├── test_io_media_integration.py        # IO and media integration tests
├── test_multimodal.py                  # General multimodal functionality tests
├── vision_extraction/
│   └── test_extraction.py              # Vision extraction specific tests
├── audio_extraction/
│   └── test_audio_extraction.py        # Audio extraction specific tests
└── multimodal_extraction/
    └── test_multimodal.py               # Multimodal extraction specific tests
```

## 🧪 Test Categories

### 1. **Dataset Standardization Tests** (`test_dataset_standardization.py`)
- Validates all input datasets follow the standard format
- Checks for required fields: `id`, `text`, `image_path`, `audio_path`
- Verifies data types and field consistency
- Analyzes modality coverage across datasets
- Checks for legacy field usage in code

### 2. **IO & Media Integration Tests** (`test_io_media_integration.py`)
- Tests `io.py` JSON operations (read/write)
- Tests `media.py` audio and image functions
- Validates media URL checking and file validation
- Tests cross-example integration consistency
- Verifies error handling and fallback mechanisms

### 3. **General Multimodal Tests** (`test_multimodal.py`)
- Tests core multimodal functionality
- Schema validation across all modality combinations
- Modality detection and routing
- Cross-modal analysis capabilities
- Basic extraction workflows

### 4. **Vision Extraction Tests** (`vision_extraction/test_extraction.py`)
- Product extraction from images
- Vision-only analysis workflows
- Image validation and processing
- Schema compliance for vision outputs

### 5. **Audio Extraction Tests** (`audio_extraction/test_audio_extraction.py`)
- Audio analysis and transcription
- Audio-only analysis workflows
- URL and local file handling
- Schema compliance for audio outputs

### 6. **Multimodal Extraction Tests** (`multimodal_extraction/test_multimodal.py`)
- Complete multimodal analysis (text + vision + audio)
- Cross-modal alignment and contradiction detection
- All modality combination testing
- Advanced multimodal schema validation

## 🚀 Running Tests

### Run All Example Tests
```bash
# From the root directory
python tests/examples/run_example_tests.py
```

### Run Individual Test Categories
```bash
# Dataset standardization
python tests/examples/test_dataset_standardization.py

# IO & Media integration
python tests/examples/test_io_media_integration.py

# Vision extraction
python tests/examples/vision_extraction/test_extraction.py

# Audio extraction
python tests/examples/audio_extraction/test_audio_extraction.py

# Multimodal extraction
python tests/examples/multimodal_extraction/test_multimodal.py
```

### Run with Main Test Suite
```bash
# Run all tests including examples
python tests/run_all_tests.py
```

## 📊 Test Coverage

### Core Functionality
- ✅ **Dataset Format Validation**: All datasets follow standard format
- ✅ **IO Operations**: JSON read/write with `io.py`
- ✅ **Media Processing**: Image and audio handling with `media.py`
- ✅ **Schema Validation**: All output schemas properly validated
- ✅ **Error Handling**: Graceful error handling and recovery

### Modality Combinations
- ✅ **Text Only**: Pure text analysis
- ✅ **Vision Only**: Image-only analysis
- ✅ **Audio Only**: Audio-only analysis
- ✅ **Text + Vision**: Combined text and image analysis
- ✅ **Text + Audio**: Combined text and audio analysis
- ✅ **Vision + Audio**: Combined image and audio analysis
- ✅ **Text + Vision + Audio**: Complete multimodal analysis

### Integration Points
- ✅ **Cross-Example Consistency**: Unified interfaces across examples
- ✅ **Media Validation**: Consistent media handling
- ✅ **Prompt Building**: Standardized prompt construction
- ✅ **Result Processing**: Consistent output formatting

## 🔧 Test Requirements

### Dependencies
- `some` library (all modules)
- `instructor` library (for structured outputs)
- OpenAI API access (for model testing)
- Internet connection (for remote audio URLs)

### Optional Assets
- `rdj.jpg` in root directory (for vision tests)
- Local audio files (tests work with remote URLs as fallback)

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 📈 Test Metrics

Each test category provides detailed metrics:

### Success Metrics
- **Pass Rate**: Percentage of tests passing
- **Coverage**: Modality combinations tested
- **Performance**: Execution time and token usage
- **Quality**: Schema compliance and validation

### Failure Analysis
- **Error Types**: Categorized failure reasons
- **Recovery**: Graceful degradation testing
- **Edge Cases**: Boundary condition handling

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running from the root directory
   - Check that `some` package is properly installed

2. **API Errors**
   - Verify OpenAI API key is set
   - Check internet connection for remote resources

3. **File Not Found**
   - Some tests require `rdj.jpg` in root directory
   - Audio tests use remote URLs as fallback

4. **Schema Validation Errors**
   - Check that all required fields are present
   - Verify data types match schema definitions

### Debug Mode
Add `--verbose` flag or set environment variable:
```bash
export SOME_TEST_VERBOSE=1
python tests/examples/run_example_tests.py
```

## 🎯 Test Philosophy

### Comprehensive Coverage
- **All Examples**: Every extraction example has dedicated tests
- **All Modalities**: Every modality combination is tested
- **All Integrations**: Cross-example consistency verified

### Real-World Scenarios
- **Actual Data**: Tests use realistic sample data
- **Error Conditions**: Tests handle missing files, invalid URLs
- **Performance**: Tests measure actual execution time and costs

### Maintainability
- **Standardized Format**: All datasets follow the same structure
- **Consistent APIs**: Unified interfaces across all examples
- **Clear Documentation**: Every test is well-documented

## 🔄 Continuous Integration

These tests are designed to be run in CI/CD pipelines:

- **Fast Execution**: Most tests complete in under 2 minutes
- **Reliable**: Graceful handling of missing resources
- **Informative**: Clear pass/fail reporting with detailed metrics
- **Isolated**: Tests don't interfere with each other

## 📚 Related Documentation

- [Vision Extraction Example](../../some/examples/vision_extraction/README.md)
- [Audio Extraction Example](../../some/examples/audio_extraction/README.md)
- [Multimodal Extraction Example](../../some/examples/multimodal_extraction/README.md)
- [Dataset Standardization Guide](../../some/examples/INTEGRATION_SUMMARY.md)

The example tests provide comprehensive validation of all extraction capabilities, ensuring reliability and consistency across the entire `some` library ecosystem.

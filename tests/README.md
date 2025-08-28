# SOME Package Unit Tests

This directory contains comprehensive unit tests for all modules in the some package.

## Test Files

- `__init___test.py` - Tests for the package initialization and lazy loading
- `cli_test.py` - Tests for CLI functionality (now disabled)
- `inference_test.py` - Tests for language model inference and providers
- `io_test.py` - Tests for file I/O operations (JSON and text)
- `main_test.py` - Tests for utility functions in main module
- `metrics_test.py` - Tests for metrics collection and reporting
- `progress_test.py` - Tests for progress bar utilities
- `prompting_test.py` - Tests for prompt builder base class

## Running Tests

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Specific Test Module
```bash
cd tests
python run_all_tests.py inference_test
```

### Run Tests with unittest
```bash
# From the project root
python -m unittest discover tests -p "*_test.py" -v

# Run specific test file
python -m unittest tests.inference_test -v

# Run specific test class
python -m unittest tests.inference_test.TestBaseLanguageModel -v

# Run specific test method
python -m unittest tests.inference_test.TestBaseLanguageModel.test_base_language_model_is_abstract -v
```

## Test Coverage

The tests cover:

### Core Functionality
- Abstract base classes and their implementations
- Language model providers (OpenAI, Ollama, custom)
- File I/O operations with error handling
- Metrics collection and reporting
- Progress bar utilities

### Edge Cases
- Empty inputs and outputs
- Invalid file formats
- Missing files and directories
- Unicode handling
- Error conditions and exceptions

### Integration Points
- Module imports and lazy loading
- Provider registration and discovery
- Configuration and default values

## Dependencies

The tests use Python's built-in `unittest` framework and `unittest.mock` for mocking external dependencies. No additional test dependencies are required.

## Test Structure

Each test file follows a consistent structure:
- Import statements and setup
- Test classes grouped by functionality
- Individual test methods with descriptive names
- Proper setup and teardown using `setUp()` and `tearDown()`
- Mock usage for external dependencies
- Temporary file handling for I/O tests

## Adding New Tests

When adding new functionality to the extraction package:

1. Create or update the corresponding test file
2. Follow the existing naming conventions (`*_test.py`)
3. Use descriptive test method names (`test_what_it_does`)
4. Include both positive and negative test cases
5. Mock external dependencies appropriately
6. Clean up any temporary resources

## Notes

- CLI functionality has been removed from the package, but tests remain to verify the removal
- Tests use mocking extensively to avoid dependencies on external services
- File I/O tests use temporary files to avoid affecting the file system
- All tests should be able to run independently and in any order

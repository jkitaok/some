"""
Unit tests for extraction/main.py module.

Tests utility functions for data loading, saving, and other extraction helpers.
CLI functionality has been removed.
"""
import unittest
import tempfile
import os
import json
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


class TestConfigureLogging(unittest.TestCase):
    """Test cases for configure_logging function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear existing handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_configure_logging_warning_level(self):
        """Test configure_logging with verbosity 0 (WARNING level)."""
        from extraction.main import configure_logging
        
        configure_logging(0)
        
        root = logging.getLogger()
        self.assertEqual(root.level, logging.WARNING)

    def test_configure_logging_info_level(self):
        """Test configure_logging with verbosity 1 (INFO level)."""
        from extraction.main import configure_logging
        
        configure_logging(1)
        
        root = logging.getLogger()
        self.assertEqual(root.level, logging.INFO)

    def test_configure_logging_debug_level(self):
        """Test configure_logging with verbosity 2+ (DEBUG level)."""
        from extraction.main import configure_logging
        
        configure_logging(2)
        
        root = logging.getLogger()
        self.assertEqual(root.level, logging.DEBUG)

    def test_configure_logging_removes_existing_handlers(self):
        """Test that configure_logging removes existing handlers."""
        from extraction.main import configure_logging
        
        # Add a dummy handler
        root = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root.addHandler(dummy_handler)
        
        initial_count = len(root.handlers)
        self.assertGreater(initial_count, 0)
        
        configure_logging(0)
        
        # Should have exactly one handler (the new one)
        self.assertEqual(len(root.handlers), 1)


class TestLoadDataFromFile(unittest.TestCase):
    """Test cases for load_data_from_file function."""

    def test_load_json_file_list(self):
        """Test loading JSON file containing a list."""
        from extraction.main import load_data_from_file
        
        test_data = [{"id": 1, "text": "item1"}, {"id": 2, "text": "item2"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = load_data_from_file(temp_path)
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_path)

    def test_load_json_file_dict(self):
        """Test loading JSON file containing a single dict."""
        from extraction.main import load_data_from_file
        
        test_data = {"id": 1, "text": "single item"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = load_data_from_file(temp_path)
            self.assertEqual(result, [test_data])  # Should be wrapped in list
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_file(self):
        """Test loading JSONL file."""
        from extraction.main import load_data_from_file
        
        test_data = [{"id": 1, "text": "item1"}, {"id": 2, "text": "item2"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            result = load_data_from_file(temp_path)
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_path)

    def test_load_txt_file(self):
        """Test loading text file."""
        from extraction.main import load_data_from_file
        
        test_lines = ["Line 1", "Line 2", "Line 3"]
        expected = [{"text": line} for line in test_lines]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(test_lines))
            temp_path = f.name
        
        try:
            result = load_data_from_file(temp_path)
            self.assertEqual(result, expected)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises FileNotFoundError."""
        from extraction.main import load_data_from_file
        
        with self.assertRaises(FileNotFoundError):
            load_data_from_file("nonexistent.json")

    def test_load_unsupported_format_raises_error(self):
        """Test loading unsupported file format raises ValueError."""
        from extraction.main import load_data_from_file
        
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_data_from_file(temp_path)
            self.assertIn("Unsupported file format", str(context.exception))
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json_type_raises_error(self):
        """Test loading JSON with invalid type raises ValueError."""
        from extraction.main import load_data_from_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump("just a string", f)  # Not list or dict
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_data_from_file(temp_path)
            self.assertIn("JSON file must contain a list or dict", str(context.exception))
        finally:
            os.unlink(temp_path)


class TestLoadPromptBuilderClass(unittest.TestCase):
    """Test cases for load_prompt_builder_class function."""

    def test_load_existing_class(self):
        """Test loading an existing class."""
        from extraction.main import load_prompt_builder_class
        
        # Use a built-in class for testing
        result = load_prompt_builder_class('builtins', 'dict')
        self.assertEqual(result, dict)

    def test_load_nonexistent_module_raises_error(self):
        """Test loading from non-existent module raises ImportError."""
        from extraction.main import load_prompt_builder_class
        
        with self.assertRaises(ImportError):
            load_prompt_builder_class('nonexistent_module', 'SomeClass')

    def test_load_nonexistent_class_raises_error(self):
        """Test loading non-existent class raises AttributeError."""
        from extraction.main import load_prompt_builder_class
        
        with self.assertRaises(AttributeError):
            load_prompt_builder_class('builtins', 'NonExistentClass')


if __name__ == '__main__':
    unittest.main()

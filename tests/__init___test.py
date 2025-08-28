"""
Unit tests for extraction/__init__.py module.

Tests the lazy loading functionality for BaseLanguageModel and register_language_model.
"""
import unittest
from unittest.mock import patch, MagicMock


class TestExtractionInit(unittest.TestCase):
    """Test cases for extraction package initialization."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import extraction
        expected_exports = ["BaseLanguageModel", "register_language_model"]
        self.assertEqual(extraction.__all__, expected_exports)

    @patch('extraction.inference.BaseLanguageModel')
    @patch('extraction.inference.register_language_model')
    def test_lazy_loading_base_language_model(self, mock_register, mock_base_lm):
        """Test lazy loading of BaseLanguageModel."""
        # Clear any cached imports
        import sys
        if 'extraction' in sys.modules:
            del sys.modules['extraction']
        
        import extraction
        
        # Access BaseLanguageModel - should trigger lazy loading
        result = extraction.BaseLanguageModel
        
        # Should return the mocked BaseLanguageModel
        self.assertEqual(result, mock_base_lm)

    @patch('extraction.inference.BaseLanguageModel')
    @patch('extraction.inference.register_language_model')
    def test_lazy_loading_register_language_model(self, mock_register, mock_base_lm):
        """Test lazy loading of register_language_model."""
        # Clear any cached imports
        import sys
        if 'extraction' in sys.modules:
            del sys.modules['extraction']
        
        import extraction
        
        # Access register_language_model - should trigger lazy loading
        result = extraction.register_language_model
        
        # Should return the mocked register_language_model
        self.assertEqual(result, mock_register)

    def test_invalid_attribute_raises_error(self):
        """Test that accessing invalid attributes raises AttributeError."""
        import extraction
        
        with self.assertRaises(AttributeError):
            _ = extraction.NonExistentAttribute

    def test_getattr_with_valid_names(self):
        """Test __getattr__ with valid names."""
        import extraction
        
        # Mock the inference module
        with patch('extraction.inference.BaseLanguageModel') as mock_base_lm, \
             patch('extraction.inference.register_language_model') as mock_register:
            
            # Test BaseLanguageModel
            result = extraction.__getattr__('BaseLanguageModel')
            self.assertEqual(result, mock_base_lm)
            
            # Test register_language_model
            result = extraction.__getattr__('register_language_model')
            self.assertEqual(result, mock_register)

    def test_getattr_with_invalid_name(self):
        """Test __getattr__ with invalid name raises AttributeError."""
        import extraction
        
        with self.assertRaises(AttributeError):
            extraction.__getattr__('InvalidName')


if __name__ == '__main__':
    unittest.main()

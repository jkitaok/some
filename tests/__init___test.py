"""
Unit tests for some/__init__.py module.

Tests the lazy loading functionality for BaseLanguageModel and register_language_model.
"""
import unittest
from unittest.mock import patch, MagicMock


class TestSomeInit(unittest.TestCase):
    """Test cases for some package initialization."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import some
        expected_exports = ["BaseLanguageModel", "register_language_model"]
        self.assertEqual(some.__all__, expected_exports)

    @patch('some.inference.BaseLanguageModel')
    @patch('some.inference.register_language_model')
    def test_lazy_loading_base_language_model(self, mock_register, mock_base_lm):
        """Test lazy loading of BaseLanguageModel."""
        # Clear any cached imports
        import sys
        if 'some' in sys.modules:
            del sys.modules['some']

        import some

        # Access BaseLanguageModel - should trigger lazy loading
        result = some.BaseLanguageModel

        # Should return the mocked BaseLanguageModel
        self.assertEqual(result, mock_base_lm)

    @patch('some.inference.BaseLanguageModel')
    @patch('some.inference.register_language_model')
    def test_lazy_loading_register_language_model(self, mock_register, mock_base_lm):
        """Test lazy loading of register_language_model."""
        # Clear any cached imports
        import sys
        if 'some' in sys.modules:
            del sys.modules['some']

        import some

        # Access register_language_model - should trigger lazy loading
        result = some.register_language_model

        # Should return the mocked register_language_model
        self.assertEqual(result, mock_register)

    def test_invalid_attribute_raises_error(self):
        """Test that accessing invalid attributes raises AttributeError."""
        import some

        with self.assertRaises(AttributeError):
            _ = some.NonExistentAttribute

    def test_getattr_with_valid_names(self):
        """Test __getattr__ with valid names."""
        import some

        # Mock the inference module
        with patch('some.inference.BaseLanguageModel') as mock_base_lm, \
             patch('some.inference.register_language_model') as mock_register:

            # Test BaseLanguageModel
            result = some.__getattr__('BaseLanguageModel')
            self.assertEqual(result, mock_base_lm)

            # Test register_language_model
            result = some.__getattr__('register_language_model')
            self.assertEqual(result, mock_register)

    def test_getattr_with_invalid_name(self):
        """Test __getattr__ with invalid name raises AttributeError."""
        import some

        with self.assertRaises(AttributeError):
            some.__getattr__('InvalidName')


if __name__ == '__main__':
    unittest.main()

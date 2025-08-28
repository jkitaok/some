"""
Unit tests for extraction/inference.py module.

Tests language model functionality, providers, and registry.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
import os
from typing import Dict, Any, List


class TestBaseLanguageModel(unittest.TestCase):
    """Test cases for BaseLanguageModel abstract class."""

    def test_base_language_model_is_abstract(self):
        """Test that BaseLanguageModel cannot be instantiated directly."""
        from extraction.inference import BaseLanguageModel
        
        with self.assertRaises(TypeError):
            BaseLanguageModel()

    def test_generate_method_is_abstract(self):
        """Test that generate method is abstract."""
        from extraction.inference import BaseLanguageModel
        
        # Create a concrete subclass without implementing generate
        class IncompleteModel(BaseLanguageModel):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteModel()


class TestLanguageModelRegistry(unittest.TestCase):
    """Test cases for language model registry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear registry before each test
        from extraction.inference import LANGUAGE_MODEL_REGISTRY
        LANGUAGE_MODEL_REGISTRY.clear()

    def test_register_language_model(self):
        """Test registering a language model."""
        from extraction.inference import register_language_model, LANGUAGE_MODEL_REGISTRY
        
        mock_factory = Mock()
        register_language_model("test_provider", mock_factory)
        
        self.assertIn("test_provider", LANGUAGE_MODEL_REGISTRY)
        self.assertEqual(LANGUAGE_MODEL_REGISTRY["test_provider"], mock_factory)

    def test_register_language_model_case_insensitive(self):
        """Test that registration is case insensitive."""
        from extraction.inference import register_language_model, LANGUAGE_MODEL_REGISTRY
        
        mock_factory = Mock()
        register_language_model("TEST_PROVIDER", mock_factory)
        
        self.assertIn("test_provider", LANGUAGE_MODEL_REGISTRY)

    def test_set_default_model(self):
        """Test setting default model for a provider."""
        from extraction.inference import set_default_model, DEFAULT_MODEL_REGISTRY
        
        set_default_model("openai", "gpt-4")
        self.assertEqual(DEFAULT_MODEL_REGISTRY["openai"], "gpt-4")

    def test_set_default_model_case_insensitive(self):
        """Test that set_default_model is case insensitive."""
        from extraction.inference import set_default_model, DEFAULT_MODEL_REGISTRY
        
        set_default_model("OPENAI", "gpt-4")
        self.assertEqual(DEFAULT_MODEL_REGISTRY["openai"], "gpt-4")


class TestGetLanguageModel(unittest.TestCase):
    """Test cases for get_language_model function."""

    def setUp(self):
        """Set up test fixtures."""
        from extraction.inference import LANGUAGE_MODEL_REGISTRY
        LANGUAGE_MODEL_REGISTRY.clear()

    @patch('extraction.inference.OpenAILanguageModel')
    def test_get_openai_model(self, mock_openai_class):
        """Test getting OpenAI language model."""
        from extraction.inference import get_language_model
        
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance
        
        result = get_language_model(provider="openai")
        
        mock_openai_class.assert_called_once()
        self.assertEqual(result, mock_instance)

    @patch('extraction.inference.OllamaLanguageModel')
    def test_get_ollama_model(self, mock_ollama_class):
        """Test getting Ollama language model."""
        from extraction.inference import get_language_model
        
        mock_instance = Mock()
        mock_ollama_class.return_value = mock_instance
        
        result = get_language_model(provider="ollama")
        
        mock_ollama_class.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_get_custom_model(self):
        """Test getting custom registered model."""
        from extraction.inference import get_language_model, register_language_model
        
        mock_factory = Mock()
        mock_instance = Mock()
        mock_factory.return_value = mock_instance
        
        register_language_model("custom", mock_factory)
        
        result = get_language_model(provider="custom")
        
        mock_factory.assert_called_once()
        self.assertEqual(result, mock_instance)

    def test_get_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        from extraction.inference import get_language_model
        
        with self.assertRaises(ValueError) as context:
            get_language_model(provider="unknown")
        
        self.assertIn("Unknown provider", str(context.exception))

    @patch('extraction.inference.OpenAILanguageModel')
    def test_get_model_with_kwargs(self, mock_openai_class):
        """Test getting model with additional kwargs."""
        from extraction.inference import get_language_model
        
        get_language_model(provider="openai", api_key="test_key", model="gpt-4")
        
        mock_openai_class.assert_called_once_with(api_key="test_key", model="gpt-4")


if __name__ == '__main__':
    unittest.main()

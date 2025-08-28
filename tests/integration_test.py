"""Integration tests for the new prompt builder architecture."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any
from pydantic import BaseModel

from some.prompting import BasePromptBuilder
from some.inference import OpenAILanguageModel, OllamaLanguageModel


class TestSchema(BaseModel):
    """Test schema for structured output."""
    name: str
    value: int


class TestPromptBuilder(BasePromptBuilder):
    """Test prompt builder using new format."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt_text": f"Extract information from: {item['text']}",
            "response_format": TestSchema,
            "result_key": "test_result"
        }


class TestPromptBuilderWithImage(BasePromptBuilder):
    """Test prompt builder with image using new format."""
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt_text": f"Analyze this image: {item['text']}",
            "image_path": item.get("image_path"),
            "response_format": TestSchema,
            "result_key": "test_result"
        }


class TestNewArchitecture(unittest.TestCase):
    """Test cases for the new prompt builder architecture."""

    def test_text_only_prompt_builder(self):
        """Test that text-only prompt builders work with new format."""
        builder = TestPromptBuilder()
        item = {"text": "test input"}
        
        result = builder.build(item)
        
        expected = {
            "prompt_text": "Extract information from: test input",
            "response_format": TestSchema,
            "result_key": "test_result"
        }
        
        self.assertEqual(result, expected)

    def test_image_prompt_builder(self):
        """Test that image prompt builders work with new format."""
        builder = TestPromptBuilderWithImage()
        item = {"text": "test input", "image_path": "/path/to/image.jpg"}
        
        result = builder.build(item)
        
        expected = {
            "prompt_text": "Analyze this image: test input",
            "image_path": "/path/to/image.jpg",
            "response_format": TestSchema,
            "result_key": "test_result"
        }
        
        self.assertEqual(result, expected)

    @patch('some.inference.OpenAI')
    def test_openai_build_messages_text_only(self, mock_openai):
        """Test OpenAI build_messages with text-only prompt."""
        mock_client = MagicMock()

        with patch('some.inference.os.getenv', return_value='fake_key'):
            model = OpenAILanguageModel()
            model.client = mock_client
        
        prompt_data = {
            "prompt_text": "Test prompt",
            "response_format": TestSchema,
            "result_key": "test"
        }
        
        messages = model.build_messages(prompt_data)
        
        expected = [
            {
                "role": "user",
                "content": "Test prompt"
            }
        ]
        
        self.assertEqual(messages, expected)

    @patch('some.media.encode_base64_content_from_path')
    @patch('some.media.get_image_mime_type')
    @patch('some.inference.OpenAI')
    def test_openai_build_messages_with_image(self, mock_openai, mock_mime, mock_encode):
        """Test OpenAI build_messages with image prompt."""
        mock_client = MagicMock()
        mock_encode.return_value = "fake_base64_data"
        mock_mime.return_value = "image/jpeg"
        
        with patch('some.inference.os.getenv', return_value='fake_key'):
            model = OpenAILanguageModel()
            model.client = mock_client
        
        prompt_data = {
            "prompt_text": "Analyze this image",
            "image_path": "/path/to/image.jpg",
            "response_format": TestSchema,
            "result_key": "test"
        }
        
        messages = model.build_messages(prompt_data)
        
        expected = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,fake_base64_data"
                        }
                    }
                ]
            }
        ]
        
        self.assertEqual(messages, expected)
        mock_encode.assert_called_once_with("/path/to/image.jpg")
        mock_mime.assert_called_once_with("/path/to/image.jpg")

    @patch('some.inference.OpenAI')
    def test_ollama_build_messages_text_only(self, mock_openai):
        """Test Ollama build_messages with text-only prompt."""
        mock_client = MagicMock()

        model = OllamaLanguageModel()
        model.client = mock_client
        
        prompt_data = {
            "prompt_text": "Test prompt",
            "response_format": TestSchema,
            "result_key": "test"
        }
        
        messages = model.build_messages(prompt_data)
        
        expected = [
            {
                "role": "user",
                "content": "Test prompt"
            }
        ]
        
        self.assertEqual(messages, expected)


if __name__ == '__main__':
    unittest.main()

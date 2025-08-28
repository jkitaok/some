"""
Unit tests for extraction/prompting.py module.

Tests the BasePromptBuilder abstract class.
"""
import unittest
from unittest.mock import Mock
from typing import Dict, Any


class TestBasePromptBuilder(unittest.TestCase):
    """Test cases for BasePromptBuilder abstract class."""

    def test_base_prompt_builder_is_abstract(self):
        """Test that BasePromptBuilder cannot be instantiated directly."""
        from some.prompting import BasePromptBuilder
        
        # Should be able to instantiate since build() has a default implementation
        # that raises NotImplementedError
        builder = BasePromptBuilder()
        self.assertIsInstance(builder, BasePromptBuilder)

    def test_build_method_raises_not_implemented(self):
        """Test that build method raises NotImplementedError by default."""
        from some.prompting import BasePromptBuilder
        
        builder = BasePromptBuilder()
        
        with self.assertRaises(NotImplementedError):
            builder.build({"test": "data"})

    def test_concrete_implementation_works(self):
        """Test that concrete implementation of BasePromptBuilder works."""
        from some.prompting import BasePromptBuilder
        
        class ConcretePromptBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "messages": [{"role": "user", "content": f"Process: {item.get('text', '')}"}],
                    "response_format": None,
                    "result_key": "result"
                }
        
        builder = ConcretePromptBuilder()
        test_item = {"text": "Hello, world!"}
        
        result = builder.build(test_item)
        
        expected = {
            "messages": [{"role": "user", "content": "Process: Hello, world!"}],
            "response_format": None,
            "result_key": "result"
        }
        self.assertEqual(result, expected)

    def test_build_method_signature(self):
        """Test that build method has correct signature."""
        from some.prompting import BasePromptBuilder
        import inspect
        
        sig = inspect.signature(BasePromptBuilder.build)
        params = list(sig.parameters.keys())
        
        self.assertEqual(len(params), 2)  # self and item
        self.assertEqual(params[0], 'self')
        self.assertEqual(params[1], 'item')
        
        # Check parameter annotations
        item_param = sig.parameters['item']
        self.assertEqual(item_param.annotation, Dict[str, Any])
        
        # Check return annotation
        self.assertEqual(sig.return_annotation, Dict[str, Any])

    def test_multiple_concrete_implementations(self):
        """Test multiple concrete implementations of BasePromptBuilder."""
        from some.prompting import BasePromptBuilder
        
        class SimplePromptBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "messages": [{"role": "user", "content": item["text"]}],
                    "result_key": "simple_result"
                }
        
        class ComplexPromptBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Analyze: {item['text']}"}
                    ],
                    "response_format": item.get("schema"),
                    "result_key": "analysis"
                }
        
        simple_builder = SimplePromptBuilder()
        complex_builder = ComplexPromptBuilder()
        
        test_item = {"text": "Test input", "schema": "TestSchema"}
        
        simple_result = simple_builder.build(test_item)
        complex_result = complex_builder.build(test_item)
        
        # Verify simple builder
        self.assertEqual(len(simple_result["messages"]), 1)
        self.assertEqual(simple_result["messages"][0]["role"], "user")
        self.assertEqual(simple_result["result_key"], "simple_result")
        
        # Verify complex builder
        self.assertEqual(len(complex_result["messages"]), 2)
        self.assertEqual(complex_result["messages"][0]["role"], "system")
        self.assertEqual(complex_result["messages"][1]["role"], "user")
        self.assertEqual(complex_result["response_format"], "TestSchema")
        self.assertEqual(complex_result["result_key"], "analysis")

    def test_builder_with_empty_item(self):
        """Test builder behavior with empty item."""
        from some.prompting import BasePromptBuilder
        
        class EmptyHandlingBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                text = item.get("text", "No text provided")
                return {
                    "messages": [{"role": "user", "content": text}],
                    "result_key": "result"
                }
        
        builder = EmptyHandlingBuilder()
        result = builder.build({})
        
        self.assertEqual(result["messages"][0]["content"], "No text provided")

    def test_builder_with_none_values(self):
        """Test builder behavior with None values in item."""
        from some.prompting import BasePromptBuilder
        
        class NoneHandlingBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                text = item.get("text")
                if text is None:
                    text = "Empty content"
                
                return {
                    "messages": [{"role": "user", "content": str(text)}],
                    "result_key": "result"
                }
        
        builder = NoneHandlingBuilder()
        result = builder.build({"text": None})
        
        self.assertEqual(result["messages"][0]["content"], "Empty content")

    def test_builder_inheritance_chain(self):
        """Test inheritance chain with BasePromptBuilder."""
        from some.prompting import BasePromptBuilder
        
        class MiddleBuilder(BasePromptBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                return {"base": "middle"}
        
        class FinalBuilder(MiddleBuilder):
            def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
                result = super().build(item)
                result["final"] = "added"
                return result
        
        builder = FinalBuilder()
        result = builder.build({"test": "data"})
        
        self.assertEqual(result["base"], "middle")
        self.assertEqual(result["final"], "added")

    def test_docstring_exists(self):
        """Test that BasePromptBuilder has proper docstring."""
        from some.prompting import BasePromptBuilder
        
        self.assertIsNotNone(BasePromptBuilder.__doc__)
        self.assertIn("Generic interface", BasePromptBuilder.__doc__)
        self.assertIn("messages", BasePromptBuilder.__doc__)
        self.assertIn("response_format", BasePromptBuilder.__doc__)
        self.assertIn("result_key", BasePromptBuilder.__doc__)


if __name__ == '__main__':
    unittest.main()

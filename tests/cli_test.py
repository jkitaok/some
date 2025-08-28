"""
Unit tests for extraction/cli.py module.

Tests the CLI functionality (which has been disabled).
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO


class TestCLI(unittest.TestCase):
    """Test cases for CLI module."""

    @patch('sys.exit')
    @patch('builtins.print')
    def test_main_shows_removal_message(self, mock_print, mock_exit):
        """Test that main() shows CLI removal message and exits."""
        from extraction.cli import main
        
        main([])
        
        # Should print removal messages
        expected_calls = [
            unittest.mock.call("CLI functionality has been removed from this package."),
            unittest.mock.call("Please use the extraction modules directly in your Python code."),
            unittest.mock.call("Example:"),
            unittest.mock.call("  from extraction.inference import get_language_model"),
            unittest.mock.call("  from extraction.prompting import BasePromptBuilder")
        ]
        mock_print.assert_has_calls(expected_calls)
        
        # Should exit with code 1
        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_main_with_argv_none(self, mock_print, mock_exit):
        """Test main() with argv=None uses sys.argv."""
        from extraction.cli import main
        
        with patch('sys.argv', ['cli.py', 'some', 'args']):
            main(None)
        
        # Should still show removal message regardless of args
        mock_print.assert_called()
        mock_exit.assert_called_once_with(1)

    @patch('sys.exit')
    @patch('builtins.print')
    def test_main_with_custom_argv(self, mock_print, mock_exit):
        """Test main() with custom argv."""
        from extraction.cli import main
        
        main(['custom', 'args'])
        
        # Should show removal message regardless of args
        mock_print.assert_called()
        mock_exit.assert_called_once_with(1)

    @patch('extraction.cli.main')
    def test_main_module_execution(self, mock_main):
        """Test that __main__ execution calls main()."""
        # This tests the if __name__ == "__main__": block
        # We can't easily test this directly, but we can verify the function exists
        from extraction import cli
        self.assertTrue(hasattr(cli, 'main'))
        self.assertTrue(callable(cli.main))

    def test_main_function_signature(self):
        """Test that main function has correct signature."""
        from extraction.cli import main
        import inspect
        
        sig = inspect.signature(main)
        params = list(sig.parameters.keys())
        
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0], 'argv')
        
        # Check default value
        param = sig.parameters['argv']
        self.assertEqual(param.default, None)

    def test_imports(self):
        """Test that required imports are available."""
        from extraction import cli
        
        # Check that the module imports successfully
        self.assertTrue(hasattr(cli, 'sys'))
        self.assertTrue(hasattr(cli, 'List'))
        self.assertTrue(hasattr(cli, 'main'))


if __name__ == '__main__':
    unittest.main()

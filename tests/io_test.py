"""
Unit tests for extraction/io.py module.

Tests file I/O functionality for JSON and text files.
"""
import unittest
import tempfile
import os
import json
from unittest.mock import patch, mock_open


class TestReadJson(unittest.TestCase):
    """Test cases for read_json function."""

    def test_read_existing_json_file(self):
        """Test reading an existing JSON file."""
        from some.io import read_json
        
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = read_json(temp_path)
            self.assertEqual(result, test_data)
        finally:
            os.unlink(temp_path)

    def test_read_nonexistent_file_returns_default(self):
        """Test reading non-existent file returns default value."""
        from some.io import read_json
        
        result = read_json("nonexistent.json", default="default_value")
        self.assertEqual(result, "default_value")

    def test_read_nonexistent_file_returns_none(self):
        """Test reading non-existent file returns None when no default."""
        from some.io import read_json
        
        result = read_json("nonexistent.json")
        self.assertIsNone(result)

    def test_read_invalid_json_returns_default(self):
        """Test reading invalid JSON returns default value."""
        from some.io import read_json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            result = read_json(temp_path, default="default_value")
            self.assertEqual(result, "default_value")
        finally:
            os.unlink(temp_path)

    def test_read_empty_file_returns_default(self):
        """Test reading empty file returns default value."""
        from some.io import read_json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = read_json(temp_path, default="default_value")
            self.assertEqual(result, "default_value")
        finally:
            os.unlink(temp_path)


class TestWriteJson(unittest.TestCase):
    """Test cases for write_json function."""

    def test_write_json_file(self):
        """Test writing JSON to file."""
        from some.io import write_json, read_json
        
        test_data = {"key": "value", "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            write_json(temp_path, test_data)
            
            # Verify file was written correctly
            result = read_json(temp_path)
            self.assertEqual(result, test_data)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_json_creates_directories(self):
        """Test that write_json creates parent directories."""
        from some.io import write_json, read_json
        
        test_data = {"test": "data"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "nested", "test.json")
            
            write_json(nested_path, test_data)
            
            # Verify file was written and directories created
            self.assertTrue(os.path.exists(nested_path))
            result = read_json(nested_path)
            self.assertEqual(result, test_data)

    def test_write_json_unicode_handling(self):
        """Test that write_json handles Unicode correctly."""
        from some.io import write_json, read_json
        
        test_data = {"unicode": "测试", "emoji": "🚀", "accents": "café"}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            write_json(temp_path, test_data)
            
            # Verify Unicode was preserved
            result = read_json(temp_path)
            self.assertEqual(result, test_data)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestReadText(unittest.TestCase):
    """Test cases for read_text function."""

    def test_read_text_file(self):
        """Test reading a text file."""
        from some.io import read_text
        
        test_content = "Hello, World!\nThis is a test file.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = read_text(temp_path)
            # read_text strips trailing whitespace
            self.assertEqual(result, test_content.strip())
        finally:
            os.unlink(temp_path)

    def test_read_text_strips_whitespace(self):
        """Test that read_text strips trailing whitespace."""
        from some.io import read_text
        
        test_content = "  Content with whitespace  \n\n  "
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = read_text(temp_path)
            self.assertEqual(result, "  Content with whitespace")
        finally:
            os.unlink(temp_path)

    def test_read_text_unicode(self):
        """Test reading text file with Unicode content."""
        from some.io import read_text
        
        test_content = "Unicode test: 测试 🚀 café"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = read_text(temp_path)
            self.assertEqual(result, test_content)
        finally:
            os.unlink(temp_path)

    def test_read_text_empty_file(self):
        """Test reading empty text file."""
        from some.io import read_text
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            result = read_text(temp_path)
            self.assertEqual(result, "")
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()

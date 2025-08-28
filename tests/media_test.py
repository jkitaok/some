"""Tests for media utilities."""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock


class TestMediaUtilities(unittest.TestCase):
    """Test cases for media utility functions."""

    @patch('some.media.Image')
    def test_encode_base64_content_from_path(self):
        """Test base64 encoding of image files."""
        from some.media import encode_base64_content_from_path
        
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.format = 'JPEG'
        mock_image.save = MagicMock()
        
        mock_Image = MagicMock()
        mock_Image.open.return_value.__enter__.return_value = mock_image
        
        with patch('some.media.Image', mock_Image):
            with patch('some.media.BytesIO') as mock_bytesio:
                mock_buffer = MagicMock()
                mock_buffer.getvalue.return_value = b'fake_image_data'
                mock_bytesio.return_value = mock_buffer
                
                with patch('some.media.base64.b64encode') as mock_b64encode:
                    mock_b64encode.return_value = b'ZmFrZV9pbWFnZV9kYXRh'
                    
                    result = encode_base64_content_from_path('test.jpg')
                    
                    self.assertEqual(result, 'ZmFrZV9pbWFnZV9kYXRh')
                    mock_Image.open.assert_called_once_with('test.jpg')
                    mock_image.save.assert_called_once()

    @patch('some.media.Image', None)
    def test_encode_base64_content_from_path_no_pil(self):
        """Test that ImportError is raised when PIL is not available."""
        from some.media import encode_base64_content_from_path
        
        with self.assertRaises(ImportError) as context:
            encode_base64_content_from_path('test.jpg')
        
        self.assertIn('PIL (Pillow) is required', str(context.exception))

    @patch('some.media.Image')
    def test_get_image_mime_type(self):
        """Test getting MIME type for image files."""
        from some.media import get_image_mime_type
        
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.format = 'PNG'
        
        mock_Image = MagicMock()
        mock_Image.open.return_value.__enter__.return_value = mock_image
        
        with patch('some.media.Image', mock_Image):
            result = get_image_mime_type('test.png')
            
            self.assertEqual(result, 'image/png')
            mock_Image.open.assert_called_once_with('test.png')

    @patch('some.media.Image')
    def test_get_image_mime_type_unknown_format(self):
        """Test getting MIME type for unknown image format defaults to jpeg."""
        from some.media import get_image_mime_type
        
        # Mock PIL Image with unknown format
        mock_image = MagicMock()
        mock_image.format = 'UNKNOWN'
        
        mock_Image = MagicMock()
        mock_Image.open.return_value.__enter__.return_value = mock_image
        
        with patch('some.media.Image', mock_Image):
            result = get_image_mime_type('test.unknown')
            
            self.assertEqual(result, 'image/jpeg')  # default fallback

    @patch('some.media.Image', None)
    def test_get_image_mime_type_no_pil(self):
        """Test that ImportError is raised when PIL is not available."""
        from some.media import get_image_mime_type
        
        with self.assertRaises(ImportError) as context:
            get_image_mime_type('test.jpg')
        
        self.assertIn('PIL (Pillow) is required', str(context.exception))


if __name__ == '__main__':
    unittest.main()

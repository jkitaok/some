"""Media utilities for handling images and other media content."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

try:
    from PIL import Image
except ImportError:
    Image = None


def encode_base64_content_from_path(content_path: str) -> str:
    """Encode a local image file to base64 format.
    
    Args:
        content_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        ImportError: If PIL is not installed
        FileNotFoundError: If the image file doesn't exist
        Exception: If the image cannot be processed
    """
    if Image is None:
        raise ImportError("PIL (Pillow) is required for image processing. Install with: pip install Pillow")
    
    with Image.open(content_path) as img:
        buffer = BytesIO()
        img.save(buffer, format=img.format)  # preserve original format (JPEG, PNG, etc.)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_mime_type(content_path: str) -> str:
    """Get the MIME type for an image file.
    
    Args:
        content_path: Path to the image file
        
    Returns:
        MIME type string (e.g., 'image/jpeg', 'image/png')
        
    Raises:
        ImportError: If PIL is not installed
        FileNotFoundError: If the image file doesn't exist
    """
    if Image is None:
        raise ImportError("PIL (Pillow) is required for image processing. Install with: pip install Pillow")
    
    with Image.open(content_path) as img:
        format_to_mime = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'GIF': 'image/gif',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
            'TIFF': 'image/tiff'
        }
        return format_to_mime.get(img.format, 'image/jpeg')  # default to jpeg

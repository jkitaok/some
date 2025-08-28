"""Media utilities for handling images, audio, and other media content."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union
import urllib.request
import urllib.parse

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import wave
    import struct
except ImportError:
    wave = None
    struct = None


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
        return format_to_mime.get(img.format or 'JPEG', 'image/jpeg')  # default to jpeg


def get_image_info(image_path: str) -> dict:
    """Get basic information about an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image information (width, height, format, file_size, etc.)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If the image cannot be analyzed
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    info = {
        'file_path': image_path,
        'file_size': os.path.getsize(image_path),
        'mime_type': get_image_mime_type(image_path)
    }

    # Try to get image dimensions and format if PIL is available
    if Image is not None:
        try:
            with Image.open(image_path) as img:
                info.update({
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                })
        except Exception:
            # If we can't read image info, just continue with basic info
            pass

    return info


def encode_base64_audio_from_path(audio_path: str) -> str:
    """Encode a local audio file to base64 format.

    Args:
        audio_path: Path to the audio file

    Returns:
        Base64 encoded string of the audio file

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If the audio cannot be processed
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode('utf-8')


def get_audio_mime_type(audio_path: str) -> str:
    """Get the MIME type for an audio file based on its extension.

    Args:
        audio_path: Path to the audio file

    Returns:
        MIME type string (e.g., 'audio/wav', 'audio/mpeg')
    """
    extension = os.path.splitext(audio_path)[1].lower()

    extension_to_mime = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.wma': 'audio/x-ms-wma'
    }

    return extension_to_mime.get(extension, 'audio/wav')  # default to wav


def get_audio_info(audio_path: str) -> dict:
    """Get basic information about an audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        Dictionary with audio information (duration, channels, sample_rate, etc.)

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If the audio cannot be analyzed
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    info = {
        'file_path': audio_path,
        'file_size': os.path.getsize(audio_path),
        'mime_type': get_audio_mime_type(audio_path)
    }

    # Try to get WAV file info if it's a WAV file
    if audio_path.lower().endswith('.wav') and wave is not None:
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                info.update({
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'sample_rate': wav_file.getframerate(),
                    'frames': wav_file.getnframes(),
                    'duration_seconds': wav_file.getnframes() / wav_file.getframerate()
                })
        except Exception:
            # If we can't read WAV info, just continue with basic info
            pass

    return info


def download_media_from_url(url: str, local_path: str) -> str:
    """Download media file from URL to local path.

    Args:
        url: URL to download from
        local_path: Local path to save the file

    Returns:
        Local path where file was saved

    Raises:
        Exception: If download fails
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        urllib.request.urlretrieve(url, local_path)

        return local_path
    except Exception as e:
        raise Exception(f"Failed to download media from {url}: {e}")


def is_valid_media_url(url: str) -> bool:
    """Check if a URL appears to be a valid media URL.

    Args:
        url: URL to check

    Returns:
        True if URL appears valid for media content
    """
    try:
        parsed = urllib.parse.urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def get_media_type_from_path(file_path: str) -> str:
    """Determine media type from file path.

    Args:
        file_path: Path to the media file

    Returns:
        Media type: 'image', 'audio', or 'unknown'
    """
    extension = os.path.splitext(file_path)[1].lower()

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wma'}

    if extension in image_extensions:
        return 'image'
    elif extension in audio_extensions:
        return 'audio'
    else:
        return 'unknown'


def validate_media_input(item: dict) -> dict:
    """Validate and enhance media input data for extraction examples.

    Args:
        item: Input item dictionary that may contain media paths/URLs

    Returns:
        Enhanced item dictionary with validation and media info

    Raises:
        FileNotFoundError: If local media files don't exist
        ValueError: If URLs are invalid
    """
    enhanced_item = item.copy()
    media_info = {}

    # Validate image sources
    if 'image_path' in item:
        image_path = item['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            img_info = get_image_info(image_path)
            media_info['image_dimensions'] = f"{img_info.get('width', 'unknown')}x{img_info.get('height', 'unknown')}"
            media_info['image_size'] = img_info.get('file_size', 0)
        except Exception:
            pass
    elif 'image_url' in item:
        if not is_valid_media_url(item['image_url']):
            raise ValueError(f"Invalid image URL: {item['image_url']}")

    # Validate audio sources
    if 'audio_path' in item:
        audio_path = item['audio_path']
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            audio_info = get_audio_info(audio_path)
            if 'duration_seconds' in audio_info:
                media_info['audio_duration'] = f"{audio_info['duration_seconds']:.1f}s"
            media_info['audio_size'] = audio_info.get('file_size', 0)
        except Exception:
            pass
    elif 'audio_url' in item:
        if not is_valid_media_url(item['audio_url']):
            raise ValueError(f"Invalid audio URL: {item['audio_url']}")

    # Add media info to context if available
    if media_info:
        context = enhanced_item.get('context', '')
        media_context = ', '.join([f"{k}: {v}" for k, v in media_info.items()])
        enhanced_item['context'] = f"{context}\nMedia info: {media_context}".strip()

    return enhanced_item


def get_supported_media_extensions() -> dict:
    """Get dictionary of supported media extensions by type.

    Returns:
        Dictionary with 'image' and 'audio' keys containing lists of extensions
    """
    return {
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'],
        'audio': ['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wma']
    }

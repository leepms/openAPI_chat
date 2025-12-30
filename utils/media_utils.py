"""Utility functions for OpenAI compatible API client"""

import base64
import mimetypes
from pathlib import Path
from typing import List, Union
import logging

from ..schema import (
    ChatMessage,
    MessageContent,
    MessageContentText,
    MessageContentImageUrl,
    MessageImageUrl,
    MessageContentVideoUrl,
    MessageVideoUrl,
)
from ..exceptions import MediaProcessingError

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 data URI
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded data URI string
        
    Raises:
        MediaProcessingError: If file not found or encoding fails
    """
    try:
        path = Path(image_path)
        if not path.exists():
            raise MediaProcessingError(
                f"Image file not found: {image_path}",
                file_path=image_path,
                media_type="image"
            )
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Default to common image types
            extension = path.suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
            }
            mime_type = mime_type_map.get(extension, 'image/jpeg')
        
        # Read and encode image
        with open(path, 'rb') as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
        
        return f"data:{mime_type};base64,{base64_data}"
    
    except MediaProcessingError:
        raise
    except Exception as e:
        raise MediaProcessingError(
            f"Failed to encode image: {str(e)}",
            file_path=image_path,
            media_type="image"
        )


def encode_video_to_base64(video_path: str) -> str:
    """
    Encode video file to base64 data URI
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Base64 encoded data URI string
        
    Raises:
        MediaProcessingError: If file not found or encoding fails
    """
    try:
        path = Path(video_path)
        if not path.exists():
            raise MediaProcessingError(
                f"Video file not found: {video_path}",
                file_path=video_path,
                media_type="video"
            )
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Default to common video types
            extension = path.suffix.lower()
            mime_type_map = {
                '.mp4': 'video/mp4',
                '.webm': 'video/webm',
                '.ogg': 'video/ogg',
                '.mov': 'video/quicktime',
                '.avi': 'video/x-msvideo',
            }
            mime_type = mime_type_map.get(extension, 'video/mp4')
        
        # Read and encode video
        with open(path, 'rb') as video_file:
            video_data = video_file.read()
            base64_data = base64.b64encode(video_data).decode('utf-8')
        
        logger.info(f"Encoded video: {video_path} ({len(video_data)} bytes)")
        return f"data:{mime_type};base64,{base64_data}"
    
    except MediaProcessingError:
        raise
    except Exception as e:
        raise MediaProcessingError(
            f"Failed to encode video: {str(e)}",
            file_path=video_path,
            media_type="video"
        )


def create_text_content(text: str) -> MessageContentText:
    """
    Create text content object
    
    Args:
        text: Text string
        
    Returns:
        MessageContentText object
    """
    return MessageContentText(type="text", text=text)


def create_image_content(
    image_path: str,
    detail: str = "auto"
) -> MessageContentImageUrl:
    """
    Create image content object from image path
    
    Args:
        image_path: Path to the image file
        detail: Image detail level ("auto", "low", or "high")
        
    Returns:
        MessageContentImageUrl object
    """
    base64_uri = encode_image_to_base64(image_path)
    return MessageContentImageUrl(
        type="image_url",
        image_url=MessageImageUrl(url=base64_uri, detail=detail)
    )


def create_video_content(
    video_path: str,
    detail: str = "auto"
) -> MessageContentImageUrl:
    """
    Create video content object from video path
    
    Note: Uses image_url type as many APIs treat video similarly to images
    
    Args:
        video_path: Path to the video file
        detail: Video detail level ("auto", "low", or "high")
        
    Returns:
        MessageContentImageUrl object with video data
    """
    base64_uri = encode_video_to_base64(video_path)
    # Use explicit video type so endpoints can distinguish video vs image
    return MessageContentVideoUrl(
        type="video_url",
        video_url=MessageVideoUrl(url=base64_uri, detail=detail)
    )


def create_user_message(
    text: str,
    image_paths: Union[str, List[str], None] = None,
    video_paths: Union[str, List[str], None] = None,
    image_detail: str = "auto",
    video_detail: str = "auto"
) -> ChatMessage:
    """
    Create user message with text and optional images/videos
    
    Args:
        text: Text content
        image_paths: Single image path or list of image paths
        video_paths: Single video path or list of video paths
        image_detail: Image detail level for all images
        video_detail: Video detail level for all videos
        
    Returns:
        ChatMessage object
    """
    if image_paths is None and video_paths is None:
        # Text only message
        return ChatMessage(role="user", content=text)
    
    # Build content list with text and media
    content_list: List[MessageContent] = [create_text_content(text)]
    
    # Add images
    if image_paths is not None:
        # Normalize image_paths to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        for image_path in image_paths:
            content_list.append(create_image_content(image_path, image_detail))
    
    # Add videos
    if video_paths is not None:
        # Normalize video_paths to list
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        for video_path in video_paths:
            content_list.append(create_video_content(video_path, video_detail))
    
    return ChatMessage(role="user", content=content_list)


def create_system_message(text: str) -> ChatMessage:
    """
    Create system message
    
    Args:
        text: System prompt text
        
    Returns:
        ChatMessage object
    """
    return ChatMessage(role="system", content=text)


def create_assistant_message(text: str) -> ChatMessage:
    """
    Create assistant message
    
    Args:
        text: Assistant response text
        
    Returns:
        ChatMessage object
    """
    return ChatMessage(role="assistant", content=text)

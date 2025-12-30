"""Utilities package for OpenAI Chat API"""

from .config_loader import (
    load_config_from_yaml,
    add_config_args,
    parse_args_to_config,
    get_default_config_paths,
)

from .media_utils import (
    encode_image_to_base64,
    encode_video_to_base64,
    create_text_content,
    create_image_content,
    create_video_content,
    create_user_message,
    create_system_message,
)

__all__ = [
    # Config utilities
    'load_config_from_yaml',
    'add_config_args',
    'parse_args_to_config',
    'get_default_config_paths',
    
    # Media utilities
    'encode_image_to_base64',
    'encode_video_to_base64',
    'create_text_content',
    'create_image_content',
    'create_video_content',
    'create_user_message',
    'create_system_message',
]

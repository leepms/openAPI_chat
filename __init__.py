"""OpenAI Compatible Chat API Client"""

__version__ = "0.3.1"

from .chat_agent import ChatAgent
from .model_config import ModelConfig, ChatConfig  # ChatConfig for backward compatibility
from .runtime_config import RuntimeConfig, UsageStats
from .model_manager import ModelManager, ModelInfo
from .exceptions import (
    ChatAPIException,
    ConfigurationError,
    APIConnectionError,
    APIResponseError,
    ToolExecutionError,
    MediaProcessingError,
    ModelNotFoundError,
    TokenLimitError,
)
from .schema import (
    Tool,
    FunctionDefinition,
    ToolCall,
    FunctionCall,
    ChatMessage,
)

# Tool support
from .tools import (
    load_tools_for_agent,
    load_builtin_tools,
    ToolLoader,
)

__all__ = [
    # Core classes
    "ChatAgent",
    "ModelManager",
    
    # Configurations
    "ModelConfig",
    "ChatConfig",  # Backward compatibility
    "RuntimeConfig",
    "UsageStats",
    
    # Schema
    "Tool",
    "FunctionDefinition",
    "ToolCall",
    "FunctionCall",
    "ChatMessage",
    
    # Exceptions
    "ChatAPIException",
    "ConfigurationError",
    "APIConnectionError",
    "APIResponseError",
    "ToolExecutionError",
    "MediaProcessingError",
    "ModelNotFoundError",
    "TokenLimitError",
    
    # Model Info
    "ModelInfo",
    
    # Tools
    "load_tools_for_agent",
    "load_builtin_tools",
    "ToolLoader",
]




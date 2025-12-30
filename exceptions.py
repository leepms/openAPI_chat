"""Custom exceptions for OpenAI Chat API Client"""


class ChatAPIException(Exception):
    """Base exception for chat API errors"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            detail_str = "\n".join(f"  {k}: {v}" for k, v in self.details.items())
            return f"{self.message}\nDetails:\n{detail_str}"
        return self.message


class ConfigurationError(ChatAPIException):
    """Configuration validation error"""
    pass


class APIConnectionError(ChatAPIException):
    """API connection error"""
    
    def __init__(self, message: str, url: str = None, status_code: int = None):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class APIResponseError(ChatAPIException):
    """API response parsing error"""
    
    def __init__(self, message: str, response_data: str = None):
        details = {}
        if response_data:
            details["response"] = response_data[:500]  # Limit length
        super().__init__(message, details)


class ToolExecutionError(ChatAPIException):
    """Tool execution error"""
    
    def __init__(self, message: str, tool_name: str = None, error: Exception = None):
        details = {}
        if tool_name:
            details["tool"] = tool_name
        if error:
            details["error"] = str(error)
        super().__init__(message, details)


class MediaProcessingError(ChatAPIException):
    """Media (image/video) processing error"""
    
    def __init__(self, message: str, file_path: str = None, media_type: str = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if media_type:
            details["media_type"] = media_type
        super().__init__(message, details)


class ModelNotFoundError(ChatAPIException):
    """Model not found error"""
    
    def __init__(self, message: str, model_name: str = None, available_models: list = None):
        details = {}
        if model_name:
            details["requested_model"] = model_name
        if available_models:
            details["available_models"] = ", ".join(available_models[:10])
        super().__init__(message, details)


class TokenLimitError(ChatAPIException):
    """Token limit exceeded error"""
    
    def __init__(self, message: str, used_tokens: int = None, max_tokens: int = None):
        details = {}
        if used_tokens:
            details["used_tokens"] = used_tokens
        if max_tokens:
            details["max_tokens"] = max_tokens
        super().__init__(message, details)

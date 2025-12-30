"""Schema definitions for OpenAI compatible API"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class MessageImageUrl:
    """Image URL content"""
    url: str
    detail: str = "auto"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MessageVideoUrl:
    """Video URL content"""
    url: str
    detail: str = "auto"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MessageContentText:
    """Text content"""
    type: Literal["text"] = "text"
    text: str = ""
    
    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class MessageContentImageUrl:
    """Image URL content"""
    type: Literal["image_url"] = "image_url"
    image_url: Optional[MessageImageUrl] = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "image_url": self.image_url.to_dict() if self.image_url else None
        }


@dataclass
class MessageContentVideoUrl:
    """Video URL content"""
    type: Literal["video_url"] = "video_url"
    video_url: Optional[MessageVideoUrl] = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "video_url": self.video_url.to_dict() if self.video_url else None
        }


MessageContent = Union[MessageContentText, MessageContentImageUrl]


# ============ Tool/Function Calling ============

@dataclass
class FunctionCall:
    """Function call in message"""
    name: str
    arguments: str  # JSON string
    
    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FunctionCall':
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", "{}")
        )


@dataclass
class ToolCall:
    """Tool call in message"""
    id: str
    type: str = "function"
    function: Optional[FunctionCall] = None
    
    def to_dict(self) -> dict:
        result = {"id": self.id, "type": self.type}
        if self.function:
            result["function"] = self.function.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ToolCall':
        func_data = data.get("function")
        function = FunctionCall.from_dict(func_data) if func_data else None
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=function
        )


@dataclass
class FunctionDefinition:
    """Function definition for tool"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema
    
    def to_dict(self) -> dict:
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = self.parameters
        return result


@dataclass
class Tool:
    """Tool definition"""
    type: str = "function"
    function: Optional[FunctionDefinition] = None
    
    def to_dict(self) -> dict:
        result = {"type": self.type}
        if self.function:
            result["function"] = self.function.to_dict()
        return result


# ============ Chat Message ============

@dataclass
class ChatMessage:
    """Chat message"""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None  # For function/tool messages
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages
    tool_call_id: Optional[str] = None  # For tool messages
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API request"""
        result = {"role": self.role}
        
        if self.content is not None:
            if isinstance(self.content, str):
                result["content"] = self.content
            elif isinstance(self.content, list):
                result["content"] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in self.content
                ]
            else:
                result["content"] = self.content
        
        if self.name:
            result["name"] = self.name
        
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        return result


# ============ Request ============

@dataclass
class ChatCompletionRequest:
    """Chat completion request"""
    model: str
    messages: List[ChatMessage]
    
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 1.0
    
    # Length control
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    
    # Penalties
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Other parameters
    n: int = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    
    # Advanced features
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    
    # Tools
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # "none", "auto", "required", or specific tool
    
    # Reasoning models
    reasoning_effort: Optional[str] = None
    
    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert to dictionary for API request"""
        result = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "stop": self.stop,
            "stream": self.stream,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "seed": self.seed,
            "user": self.user,
            "reasoning_effort": self.reasoning_effort,
        }
        
        if self.tools:
            result["tools"] = [tool.to_dict() for tool in self.tools]
        
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice
        
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        
        return result


# ============ Response ============

@dataclass
class ChatCompletionChoice:
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionChoice':
        """Create from dictionary"""
        msg_data = data.get("message", {})
        
        # Parse tool calls if present
        tool_calls = None
        if "tool_calls" in msg_data:
            tool_calls = [
                ToolCall.from_dict(tc) for tc in msg_data["tool_calls"]
            ]
        
        message = ChatMessage(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content"),
            name=msg_data.get("name"),
            tool_calls=tool_calls,
            tool_call_id=msg_data.get("tool_call_id")
        )
        
        return cls(
            index=data.get("index", 0),
            message=message,
            finish_reason=data.get("finish_reason"),
            logprobs=data.get("logprobs")
        )


@dataclass
class ChatCompletionUsage:
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Dict[str, Any]] = None  # For reasoning tokens
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionUsage':
        """Create from dictionary"""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            completion_tokens_details=data.get("completion_tokens_details")
        )


@dataclass
class ChatCompletionResponse:
    """Chat completion response"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionResponse':
        """Create from dictionary"""
        choices = [
            ChatCompletionChoice.from_dict(choice)
            for choice in data.get("choices", [])
        ]
        
        usage_data = data.get("usage")
        usage = ChatCompletionUsage.from_dict(usage_data) if usage_data else None
        
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint")
        )


# ============ Streaming Response ============

@dataclass
class ChatCompletionChunkDelta:
    """Stream chunk delta"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Partial tool calls
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionChunkDelta':
        """Create from dictionary"""
        return cls(
            role=data.get("role"),
            content=data.get("content"),
            tool_calls=data.get("tool_calls")
        )


@dataclass
class ChatCompletionChunkChoice:
    """Stream chunk choice"""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionChunkChoice':
        """Create from dictionary"""
        delta_data = data.get("delta", {})
        delta = ChatCompletionChunkDelta.from_dict(delta_data)
        
        return cls(
            index=data.get("index", 0),
            delta=delta,
            finish_reason=data.get("finish_reason"),
            logprobs=data.get("logprobs")
        )


@dataclass
class ChatCompletionChunk:
    """Chat completion stream chunk"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatCompletionChunk':
        """Create from dictionary"""
        choices = [
            ChatCompletionChunkChoice.from_dict(choice)
            for choice in data.get("choices", [])
        ]
        
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            system_fingerprint=data.get("system_fingerprint")
        )

"""OpenAI Compatible Chat Agent - Complete Implementation"""

import asyncio
import inspect
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union

import httpx

from .model_config import ModelConfig, ChatConfig  # ChatConfig for backward compatibility
from .runtime_config import RuntimeConfig, UsageStats
from .exceptions import (
    APIConnectionError,
    APIResponseError,
    ToolExecutionError,
    ConfigurationError,
)
from .schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Tool,
    ToolCall,
    FunctionCall,
)
from .utils.media_utils import create_user_message, create_system_message


logger = logging.getLogger(__name__)
http_logger = logging.getLogger("openai_chatapi.http")


class ChatAgent:
    """
    OpenAI compatible chat agent with full feature support
    
    Features:
    - Streaming and non-streaming responses with live display
    - Text, image, and video inputs (multimodal)
    - Tool/function calling with auto-execution
    - All OpenAI API parameters
    - Reasoning models support
    - Comprehensive error handling
    - Token usage tracking and statistics
    """

    def __init__(
        self,
        model_config: Optional[Union[ModelConfig, ChatConfig]] = None,
        runtime_config: Optional[RuntimeConfig] = None
    ):
        """
        Initialize chat agent
        
        Args:
            model_config: Model configuration (ModelConfig or ChatConfig for backward compatibility)
            runtime_config: Runtime configuration. If None, uses default.
        """
        self.model_config = model_config or ModelConfig()
        self.runtime_config = runtime_config or RuntimeConfig()
        self.messages: List[ChatMessage] = []
        self.tools: List[Tool] = []
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Statistics tracking
        self.stats = UsageStats()
        
        # Setup HTTP client
        self._client = httpx.AsyncClient(
            timeout=self.runtime_config.timeout,
            verify=self.runtime_config.verify_ssl
        )
        
        if self.runtime_config.enable_debug:
            logger.debug(f"ChatAgent initialized with model: {self.model_config.model}")
    
    # Backward compatibility property
    @property
    def config(self) -> ModelConfig:
        """Get model config (backward compatibility)"""
        return self.model_config
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close HTTP client"""
        await self._client.aclose()
        
        if self.runtime_config.enable_debug:
            logger.debug("ChatAgent closed")
    
    # ========== Statistics Methods ==========
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.stats.reset()
        if self.runtime_config.enable_debug:
            logger.debug("Statistics reset")
    
    def save_stats(self, file_path: Optional[str] = None):
        """
        Save statistics to file
        
        Args:
            file_path: File path to save stats. If None, uses config path.
        """
        if file_path is None:
            if self.runtime_config.save_token_usage_to_file:
                file_path = self.runtime_config.token_usage_file_path
            else:
                raise ValueError("No file path specified and save_token_usage_to_file is False")
        
        self.stats.save_to_file(file_path)
        if self.runtime_config.enable_debug:
            logger.debug(f"Statistics saved to {file_path}")
    
    # ========== Configuration Methods ==========
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt"""
        self.messages = [msg for msg in self.messages if msg.role != "system"]
        self.messages.insert(0, create_system_message(prompt))
    
    def add_message(self, message: ChatMessage):
        """Add message to conversation history"""
        self.messages.append(message)
    
    def clear_history(self, keep_system: bool = True):
        """Clear conversation history"""
        if keep_system:
            self.messages = [msg for msg in self.messages if msg.role == "system"]
        else:
            self.messages = []
    
    # ========== Tool Management ==========
    
    def register_tool(self, tool: Tool, handler: Optional[Callable] = None):
        """
        Register a tool for function calling
        
        Args:
            tool: Tool definition
            handler: Optional handler function (sync or async)
        """
        self.tools.append(tool)
        if handler and tool.function:
            self.tool_handlers[tool.function.name] = handler
    
    def clear_tools(self):
        """Clear all registered tools"""
        self.tools = []
        self.tool_handlers = {}
    
    async def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """
        Execute a tool call
        
        Args:
            tool_call: ToolCall to execute
            
        Returns:
            Tool execution result as JSON string
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        if not tool_call.function:
            raise ToolExecutionError("No function in tool call")
        
        func_name = tool_call.function.name
        handler = self.tool_handlers.get(func_name)
        
        if not handler:
            raise ToolExecutionError(
                f"No handler registered for function",
                tool_name=func_name
            )
        
        try:
            # Parse arguments
            args = json.loads(tool_call.function.arguments)
            
            if self.runtime_config.enable_debug:
                logger.debug(f"Executing tool: {func_name} with args: {args}")
            
            # Execute handler (sync or async)
            if inspect.iscoroutinefunction(handler):
                result = await handler(**args)
            else:
                result = handler(**args)
            
            # Convert result to JSON string
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result)
        
        except json.JSONDecodeError as e:
            raise ToolExecutionError(
                f"Failed to parse tool arguments",
                tool_name=func_name,
                error=e
            )
        except Exception as e:
            logger.error(f"Tool execution error in {func_name}: {e}")
            raise ToolExecutionError(
                f"Tool execution failed",
                tool_name=func_name,
                error=e
            )
    
    # ========== Request Building ==========
    
    def _build_request(
        self,
        messages: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> ChatCompletionRequest:
        """Build chat completion request"""
        request_messages = messages if messages is not None else self.messages
        
        # Get tools from kwargs or use registered tools
        tools = kwargs.pop("tools", None)
        if tools is None and self.tools:
            tools = self.tools
        
        config = self.model_config
        
        return ChatCompletionRequest(
            model=kwargs.get("model", config.model),
            messages=request_messages,
            temperature=kwargs.get("temperature", config.temperature),
            top_p=kwargs.get("top_p", config.top_p),
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
            max_completion_tokens=kwargs.get("max_completion_tokens", config.max_completion_tokens),
            frequency_penalty=kwargs.get("frequency_penalty", config.frequency_penalty),
            presence_penalty=kwargs.get("presence_penalty", config.presence_penalty),
            n=kwargs.get("n", config.n),
            stop=kwargs.get("stop", config.stop),
            stream=kwargs.get("stream", config.stream),
            logprobs=kwargs.get("logprobs", config.logprobs),
            top_logprobs=kwargs.get("top_logprobs", config.top_logprobs),
            seed=kwargs.get("seed", config.seed),
            user=kwargs.get("user", config.user),
            reasoning_effort=kwargs.get("reasoning_effort", config.reasoning_effort),
            tools=tools,
            tool_choice=kwargs.get("tool_choice"),
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.model_config.api_key:
            headers["Authorization"] = f"Bearer {self.model_config.api_key}"
        return headers
    
    # ========== Chat Methods ==========
    
    async def chat(
        self,
        text: str,
        image_paths: Union[str, List[str], None] = None,
        video_paths: Union[str, List[str], None] = None,
        add_to_history: bool = True,
        auto_execute_tools: bool = True,
        max_tool_iterations: int = 5,
        **kwargs
    ) -> str:
        """
        Send chat message and get response (non-streaming)
        
        Args:
            text: Text content
            image_paths: Optional image path(s)
            video_paths: Optional video path(s)
            add_to_history: Whether to add messages to history
            auto_execute_tools: Whether to automatically execute tool calls
            max_tool_iterations: Maximum tool call iterations
            **kwargs: Additional request parameters
            
        Returns:
            Assistant response text
        """
        start_time = time.time() if self.runtime_config.capture_latency else 0
        
        try:
            user_message = create_user_message(text, image_paths, video_paths)
            
            if add_to_history:
                self.add_message(user_message)
                request_messages = None
            else:
                request_messages = self.messages + [user_message]
            
            iteration = 0
            while iteration < max_tool_iterations:
                iteration += 1
                
                # Build and send request
                request = self._build_request(messages=request_messages, stream=False, **kwargs)
                response = await self._send_request(request)
                
                assistant_message = response.choices[0].message
                
                # Track token usage
                if self.runtime_config.capture_token_usage and response.usage:
                    elapsed = time.time() - start_time if start_time else 0
                    self.stats.add_request(
                        prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(response.usage, "completion_tokens", 0),
                        latency=elapsed
                    )
                
                # Add assistant message to history
                if add_to_history:
                    self.add_message(assistant_message)
                elif request_messages is not None:
                    request_messages.append(assistant_message)
                
                # Check for tool calls
                if assistant_message.tool_calls and auto_execute_tools:
                    # Execute all tool calls
                    for tool_call in assistant_message.tool_calls:
                        result = await self._execute_tool_call(tool_call)
                        
                        # Add tool result message
                        tool_message = ChatMessage(
                            role="tool",
                            content=result,
                            tool_call_id=tool_call.id
                        )
                        
                        if add_to_history:
                            self.add_message(tool_message)
                        elif request_messages is not None:
                            request_messages.append(tool_message)
                    
                    # Continue loop to get final response
                    continue
                
                # No tool calls, return content
                return assistant_message.content or ""
            
            logger.warning(f"Reached maximum tool iterations: {max_tool_iterations}")
            return assistant_message.content or ""
        
        except Exception as e:
            # Track error
            if self.runtime_config.capture_token_usage:
                elapsed = time.time() - start_time if start_time else 0
                self.stats.add_request(latency=elapsed, error=True)
            raise
    
    async def chat_stream(
        self,
        text: str,
        image_paths: Union[str, List[str], None] = None,
        video_paths: Union[str, List[str], None] = None,
        add_to_history: bool = True,
        display_stream: bool = True,
        auto_execute_tools: bool = False,
        max_tool_iterations: int = 5,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Send chat message and get streaming response
        
        Note: Tool calling in streaming mode is complex and may not work with all models.
        For reliable tool calling, use chat() method instead.
        
        Args:
            text: Text content
            image_paths: Optional image path(s)
            video_paths: Optional video path(s)
            add_to_history: Whether to add messages to history
            display_stream: Whether to print chunks in real-time
            auto_execute_tools: Whether to automatically execute tool calls (experimental)
            max_tool_iterations: Maximum tool call iterations
            **kwargs: Additional request parameters
            
        Yields:
            Response text chunks
        """
        start_time = time.time() if self.runtime_config.capture_latency else 0
        
        try:
            user_message = create_user_message(text, image_paths, video_paths)
            
            if add_to_history:
                self.add_message(user_message)
                request_messages = None
            else:
                request_messages = self.messages + [user_message]
            
            # Build request
            request = self._build_request(messages=request_messages, stream=True, **kwargs)
            
            # Send streaming request
            full_response = []
            chunk_count = 0
            async for chunk_text in self._send_stream_request(request):
                full_response.append(chunk_text)
                chunk_count += 1
                
                # Display in real-time if enabled
                if display_stream and self.runtime_config.stream_enable_progress:
                    print(chunk_text, end='', flush=True)
                
                # Call chunk callback if configured
                if self.runtime_config.stream_chunk_callback:
                    try:
                        self.runtime_config.stream_chunk_callback(chunk_text)
                    except Exception as e:
                        logger.warning(f"Stream callback error: {e}")
                
                yield chunk_text
            
            # Print newline after streaming
            if display_stream and self.runtime_config.stream_enable_progress:
                print()
            
            if add_to_history:
                response_text = "".join(full_response)
                self.add_message(ChatMessage(role="assistant", content=response_text))
            
            # Track statistics (approximate token count for streaming)
            if self.runtime_config.capture_token_usage:
                elapsed = time.time() - start_time if start_time else 0
                # Rough estimate: ~4 chars per token
                completion_tokens = len("".join(full_response)) // 4
                self.stats.add_request(
                    completion_tokens=completion_tokens,
                    latency=elapsed
                )
            
            if self.runtime_config.enable_debug:
                logger.debug(f"Streamed {chunk_count} chunks")
        
        except Exception as e:
            # Track error
            if self.runtime_config.capture_token_usage:
                elapsed = time.time() - start_time if start_time else 0
                self.stats.add_request(latency=elapsed, error=True)
            raise
    
    # ========== HTTP Methods ==========
    
    async def _send_request(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send non-streaming request"""
        url = f"{self.model_config.api_base_url}/chat/completions"
        headers = self._get_headers()
        request_data = request.to_dict(exclude_none=True)
        # Some endpoints (e.g., qwen-compatible) require explicit control
        # of 'enable_thinking' for non-streaming calls. Ensure it's disabled
        # for non-streaming requests unless explicitly set.
        if not request_data.get("stream", False) and "enable_thinking" not in request_data:
            request_data["enable_thinking"] = False
        
        # Log HTTP request if enabled
        if self.runtime_config.capture_http_traffic:
            if self.runtime_config.log_http_requests:
                http_logger.info(f"POST {url}")
                http_logger.debug(f"Request: {json.dumps(request_data, indent=2)}")
            
            # Save to debug file if enabled
            if self.runtime_config.debug_save_requests:
                self._save_debug_data("request", request_data)
        
        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=request_data,
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Log HTTP response if enabled
            if self.runtime_config.capture_http_traffic:
                if self.runtime_config.log_http_responses:
                    http_logger.debug(f"Response: {json.dumps(response_data, indent=2)}")
                
                # Save to debug file if enabled
                if self.runtime_config.debug_save_responses:
                    self._save_debug_data("response", response_data)
            
            return ChatCompletionResponse.from_dict(response_data)
            
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            if self.runtime_config.truncate_long_errors:
                error_text = error_text[:self.runtime_config.max_error_length]
            
            logger.error(f"HTTP {e.response.status_code} error: {error_text}")
            raise APIConnectionError(
                f"API request failed with status {e.response.status_code}",
                url=url,
                status_code=e.response.status_code
            )
        
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise APIConnectionError(
                f"Failed to connect to API: {str(e)}",
                url=url
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise APIResponseError(
                "Failed to parse API response as JSON",
                response_data=str(e)
            )
        
        except Exception as e:
            logger.error(f"Unexpected request error: {e}")
            raise APIConnectionError(
                f"Request failed: {str(e)}",
                url=url
            )
    
    def _save_debug_data(self, data_type: str, data: dict):
        """Save debug data to file"""
        try:
            debug_dir = Path(self.runtime_config.debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = debug_dir / f"{data_type}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.warning(f"Failed to save debug data: {e}")
    
    async def _send_stream_request(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Send streaming request"""
        url = f"{self.model_config.api_base_url}/chat/completions"
        headers = self._get_headers()
        request_data = request.to_dict(exclude_none=True)
        
        # Log HTTP request if enabled
        if self.runtime_config.capture_http_traffic:
            if self.runtime_config.log_http_requests:
                http_logger.info(f"POST {url} (streaming)")
                http_logger.debug(f"Request: {json.dumps(request_data, indent=2)}")
            
            if self.runtime_config.debug_save_requests:
                self._save_debug_data("stream_request", request_data)
        
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=request_data,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    line = line.strip()
                    
                    if not line or line.startswith(":"):
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            chunk = ChatCompletionChunk.from_dict(chunk_data)
                            
                            if chunk.choices and len(chunk.choices) > 0:
                                choice = chunk.choices[0]
                                if choice.delta.content:
                                    yield choice.delta.content
                        
                        except json.JSONDecodeError as e:
                            if not self.runtime_config.strict_parsing:
                                logger.warning(f"Failed to parse chunk: {data[:100]}, error: {e}")
                                continue
                            else:
                                raise APIResponseError(
                                    "Failed to parse streaming chunk",
                                    response_data=data[:100]
                                )
                        except Exception as e:
                            logger.warning(f"Failed to process chunk: {e}")
                            if self.runtime_config.strict_parsing:
                                raise
                            continue
        
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            if self.runtime_config.truncate_long_errors:
                error_text = error_text[:self.runtime_config.max_error_length]
            
            logger.error(f"HTTP {e.response.status_code} error: {error_text}")
            raise APIConnectionError(
                f"Streaming request failed with status {e.response.status_code}",
                url=url,
                status_code=e.response.status_code
            )
        
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise APIConnectionError(
                f"Failed to connect to API: {str(e)}",
                url=url
            )
        
        except Exception as e:
            logger.error(f"Stream request error: {e}")
            raise APIConnectionError(
                f"Streaming request failed: {str(e)}",
                url=url
            )

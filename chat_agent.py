"""OpenAI Compatible Chat Agent - Complete Implementation"""

import asyncio
import inspect
import json
import logging
import random
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union

import httpx

from model_config import ModelConfig, ChatConfig  # ChatConfig for backward compatibility
from runtime_config import RuntimeConfig, UsageStats
from exceptions import (
    APIConnectionError,
    APIResponseError,
    ToolExecutionError,
    ConfigurationError,
)
from schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Tool,
    ToolCall,
    FunctionCall,
)
from utils.media_utils import create_user_message, create_system_message


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
        # Duplicate detection: ensure tool name (if provided) is not already registered
        if tool.function and tool.function.name:
            name = tool.function.name
            for t in self.tools:
                if t.function and t.function.name == name:
                    raise ConfigurationError(f"Tool with name '{name}' already registered")
            if name in self.tool_handlers:
                raise ConfigurationError(f"Handler for tool '{name}' already registered")

        # If handler provided, perform basic signature compatibility check
        if handler and tool.function:
            # If function.parameters looks like a JSON Schema, try to extract property names
            expected_params = set()
            params_schema = tool.function.parameters or {}
            if isinstance(params_schema, dict):
                # Common pattern: {"type":"object","properties":{...}}
                props = params_schema.get('properties') or params_schema.get('properties', {})
                if isinstance(props, dict):
                    expected_params = set(props.keys())

            # Inspect handler signature
            try:
                sig = inspect.signature(handler)
                param_names = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
                has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            except Exception:
                param_names = []
                has_var_kw = False

            # If expected_params non-empty, ensure handler can accept them (either has each param or **kwargs)
            if expected_params and not has_var_kw:
                missing = expected_params - set(param_names)
                if missing:
                    raise ConfigurationError(
                        f"Handler for '{tool.function.name}' missing parameters: {', '.join(sorted(missing))}"
                    )

            # Register handler
            self.tool_handlers[tool.function.name] = handler

        # Finally register tool metadata
        self.tools.append(tool)
    
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
            raw_args = tool_call.function.arguments if tool_call.function and tool_call.function.arguments is not None else ""

            # Try to parse JSON arguments; fall back to raw string
            parsed = None
            try:
                parsed = json.loads(raw_args)
            except Exception as e:
                if self.runtime_config.enable_debug:
                    logger.debug(f"Failed to parse JSON args: {raw_args[:100]}, error: {e}")
                parsed = None

            if self.runtime_config.enable_debug:
                logger.debug(f"Executing tool: {func_name}")
                logger.debug(f"  Raw args: {raw_args}")
                logger.debug(f"  Parsed type: {type(parsed).__name__}")
                logger.debug(f"  Parsed value: {parsed}")

            # Prepare call signature: support dict -> kwargs, list/tuple -> *args, scalar -> single positional arg, None -> no args
            call_args = []
            call_kwargs = {}
            if parsed is None:
                if raw_args:
                    # Non-JSON string, pass as single positional arg
                    call_args = [raw_args]
                else:
                    call_args = []
            elif isinstance(parsed, dict):
                call_kwargs = parsed
            elif isinstance(parsed, (list, tuple)):
                call_args = list(parsed)
            else:
                # scalar value
                call_args = [parsed]
            
            if self.runtime_config.enable_debug:
                logger.debug(f"  Call args: {call_args}")
                logger.debug(f"  Call kwargs: {call_kwargs}")

            # Execute handler (sync or async). Support handlers that return awaitables too.
            if inspect.iscoroutinefunction(handler):
                result = await handler(*call_args, **call_kwargs)
            else:
                result = handler(*call_args, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result

            # Convert result to JSON string (best-effort). If not JSON-serializable, fall back to str().
            if isinstance(result, str):
                return result
            try:
                return json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                return str(result)

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
        max_tool_iterations: Optional[int] = None,
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
            iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.runtime_config.max_tool_iterations
            while iteration < iterations_limit:
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
                        # Execute each tool with failure policy handling
                        try:
                            result = await self._execute_tool_call(tool_call)
                        except Exception as e:
                            policy = getattr(self.runtime_config, 'tool_failure_policy', 'inject_message')
                            if policy == 'raise':
                                raise
                            elif policy == 'retry_once':
                                try:
                                    result = await self._execute_tool_call(tool_call)
                                except Exception as e2:
                                    # fallback to inject message
                                    result = f"Tool execution failed: {e2}"
                            else:
                                # inject_message (default)
                                result = f"Tool execution failed: {e}"

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
                response_text = assistant_message.content or ""
                # Call response callback if configured
                if self.runtime_config.response_callback:
                    try:
                        self.runtime_config.response_callback(response_text)
                    except Exception as e:
                        logger.warning(f"Response callback error: {e}")
                return response_text
            
            logger.warning(f"Reached maximum tool iterations: {iterations_limit}")
            response_text = assistant_message.content or ""
            # Call response callback if configured
            if self.runtime_config.response_callback:
                try:
                    self.runtime_config.response_callback(response_text)
                except Exception as e:
                    logger.warning(f"Response callback error: {e}")
            return response_text
        
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
        max_tool_iterations: Optional[int] = None,
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

            iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.runtime_config.max_tool_iterations
            
            # Send streaming request and handle tool calls at the chat_stream level.
            full_response = []
            chunk_count = 0

            # Buffers for tool/function calls across chunks
            tool_buffers: Dict[str, Dict[str, str]] = {}
            tool_iterations = 0

            # Outer loop to allow restarting the stream after tool executions (like chat())
            restart_count = 0
            while True:
                restart_count += 1
                    
                inner_chunk_count = 0
                async for choice in self._send_stream_request(request):
                    chunk_count += 1
                    inner_chunk_count += 1

                    delta = choice.delta
                    
                    # Yield content if present
                    if delta.content:
                        chunk_text = delta.content
                        full_response.append(chunk_text)

                        # Display in real-time if enabled
                        if display_stream and self.runtime_config.display_stream_output:
                            print(chunk_text, end='', flush=True)

                        # Call chunk callback if configured
                        if self.runtime_config.stream_chunk_callback:
                            try:
                                self.runtime_config.stream_chunk_callback(chunk_text)
                            except Exception as e:
                                logger.warning(f"Stream callback error: {e}")

                        yield chunk_text

                    # Accumulate tool call pieces if present
                    if auto_execute_tools and delta.tool_calls:
                        for tc in delta.tool_calls:
                            # 获取tool call的标识符
                            tc_id = tc.get('id')
                            func = tc.get('function') or {}
                            name = func.get('name')
                            args_piece = func.get('arguments', '')
                            
                            # 如果有id，使用id作为key；如果只有name，使用name；都没有则使用对象id
                            if tc_id:
                                key = tc_id
                            elif name:
                                key = name
                            else:
                                # 可能是参数的后续片段，尝试找到最近的tool buffer
                                # 通常应该只有一个active的tool call
                                if tool_buffers:
                                    key = list(tool_buffers.keys())[0]
                                else:
                                    key = str(id(tc))
                            
                            buf = tool_buffers.get(key)
                            if not buf:
                                # 创建新缓冲区
                                tool_buffers[key] = {'id': tc_id or key, 'name': name or '', 'arguments': args_piece or ''}
                            else:
                                # 更新现有缓冲区
                                if name and not buf.get('name'):
                                    buf['name'] = name
                                if tc_id and not buf.get('id'):
                                    buf['id'] = tc_id
                                # 累积参数片段
                                buf['arguments'] = buf.get('arguments', '') + (args_piece or '')

                    # If finish_reason indicates function/tool call completed, execute buffered tool calls
                    finish = choice.finish_reason
                    if auto_execute_tools and finish and finish in ("function_call", "tool_call", "tool_calls", "stop"):
                        # Execute buffered tool calls (only if we have buffered calls)
                        executed_any = False
                        if tool_buffers:  # 只有当真的有缓冲的工具调用时才执行
                            # First, build the assistant message with tool_calls (required by API)
                            from schema import ToolCall as SchemaToolCall, FunctionCall as SchemaFunctionCall
                            
                            tool_calls_list = []
                            for buf_key, buf_val in tool_buffers.items():
                                func_name = buf_val.get('name', '')
                                func_args = buf_val.get('arguments', '')
                                tool_call_id = buf_val.get('id', buf_key)
                                
                                schema_func = SchemaFunctionCall(name=func_name, arguments=func_args)
                                schema_tc = SchemaToolCall(id=tool_call_id, function=schema_func)
                                tool_calls_list.append(schema_tc)
                            
                            # Create assistant message with tool calls
                            assistant_msg = ChatMessage(
                                role="assistant",
                                content=''.join(full_response) if full_response else None,
                                tool_calls=tool_calls_list
                            )
                            
                            # Add assistant message to history/request before tool messages
                            if add_to_history:
                                self.add_message(assistant_msg)
                            elif request_messages is not None:
                                request_messages.append(assistant_msg)
                            
                            # Now execute each tool and add tool result messages
                            for buf_key, buf_val in list(tool_buffers.items()):
                                if tool_iterations >= iterations_limit:
                                    logger.warning(f"Reached maximum tool iterations during streaming: {iterations_limit}")
                                    break

                                tool_iterations += 1
                                executed_any = True
                                func_name = buf_val.get('name', '')
                                func_args = buf_val.get('arguments', '')
                                tool_call_id = buf_val.get('id', buf_key)

                                if self.runtime_config.enable_debug:
                                    logger.debug(f"Executing buffered tool call: {func_name}")
                                    logger.debug(f"  Complete args: {func_args}")

                                schema_func = SchemaFunctionCall(name=func_name, arguments=func_args)
                                schema_tc = SchemaToolCall(id=tool_call_id, function=schema_func)

                                try:
                                    result = await self._execute_tool_call(schema_tc)
                                except Exception as e:
                                    policy = getattr(self.runtime_config, 'tool_failure_policy', 'inject_message')
                                    if policy == 'raise':
                                        raise
                                    elif policy == 'retry_once':
                                        try:
                                            result = await self._execute_tool_call(schema_tc)
                                        except Exception as e2:
                                            result = f"Tool execution failed: {e2}"
                                    else:
                                        result = f"Tool execution failed: {e}"

                                # Create tool message and add to history or request_messages
                                tool_message = ChatMessage(role="tool", content=result, tool_call_id=tool_call_id)
                                if add_to_history:
                                    self.add_message(tool_message)
                                elif request_messages is not None:
                                    request_messages.append(tool_message)

                                # Clear buffer for this tool call
                                tool_buffers.pop(buf_key, None)

                        # If we executed any tools, restart the stream to let the model continue (like chat())
                        if executed_any:
                            # 清空缓冲区，准备接收新的响应
                            tool_buffers.clear()
                            # Rebuild request with updated messages and continue outer while loop
                            request = self._build_request(messages=request_messages, stream=True, **kwargs)
                            # break out of the inner async for to re-open stream
                            break

                else:
                    # Normal completion of async for (no break), exit outer loop
                    break

                # If we broke from inner loop to restart stream, continue outer while to reopen new stream
                continue
            
            # Print newline after streaming
            if display_stream and self.runtime_config.display_stream_output:
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
        # of 'enable_thinking' for non-streaming calls. Only add this for
        # DashScope API to avoid incompatibility with other providers.
        if (not request_data.get("stream", False) 
            and "enable_thinking" not in request_data
            and "dashscope.aliyuncs.com" in self.model_config.api_base_url):
            request_data["enable_thinking"] = False
        
        # Log HTTP request if enabled
        if self.runtime_config.capture_http_traffic:
            if self.runtime_config.log_http_requests:
                http_logger.info(f"POST {url}")
                http_logger.debug(f"Request Headers:\n{json.dumps(dict(headers), indent=2)}")
                http_logger.debug(f"Request Body:\n{json.dumps(request_data, indent=2, ensure_ascii=False)}")
            
            # Save to debug file if enabled
            if self.runtime_config.debug_save_requests:
                self._save_debug_data("request", request_data)
        
        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.runtime_config.max_retries + 1):
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
                        http_logger.info(f"HTTP/{response.http_version} {response.status_code} {response.reason_phrase}")
                        http_logger.debug("Response Headers:")
                        for key, value in response.headers.items():
                            http_logger.debug(f"  {key}: {value}")
                        http_logger.debug(f"Response Body:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
                    
                    # Save to debug file if enabled
                    if self.runtime_config.debug_save_responses:
                        self._save_debug_data("response", {
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "body": response_data
                        })
                
                return ChatCompletionResponse.from_dict(response_data)
                
            except httpx.HTTPStatusError as e:
                # Handle rate limiting (429) and server errors (5xx) with retry
                status_code = e.response.status_code
                is_retryable = status_code == 429 or status_code >= 500
                
                if is_retryable and attempt < self.runtime_config.max_retries:
                    # Calculate exponential backoff with jitter
                    base_delay = self.runtime_config.retry_delay
                    exponential_delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, 0.1 * exponential_delay)
                    total_delay = exponential_delay + jitter
                    
                    # Check for Retry-After header (common in 429 responses)
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            total_delay = float(retry_after)
                        except ValueError:
                            pass
                    
                    logger.warning(
                        f"HTTP {status_code} error (attempt {attempt + 1}/{self.runtime_config.max_retries + 1}). "
                        f"Retrying after {total_delay:.2f}s..."
                    )
                    await asyncio.sleep(total_delay)
                    last_exception = e
                    continue
                
                # Not retryable or out of retries
                try:
                    error_text = e.response.text
                except Exception:
                    error_text = f"HTTP {status_code}"
                    
                if self.runtime_config.truncate_long_errors:
                    error_text = error_text[:self.runtime_config.max_error_length]
                
                logger.error(f"HTTP {status_code} error: {error_text}")
                raise APIConnectionError(
                    f"API request failed with status {status_code}",
                    url=url,
                    status_code=status_code
                )
            
            except httpx.ConnectError as e:
                if attempt < self.runtime_config.max_retries:
                    delay = self.runtime_config.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error (attempt {attempt + 1}/{self.runtime_config.max_retries + 1}). Retrying after {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    last_exception = e
                    continue
                    
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
        
        # If we exhausted all retries
        if last_exception:
            raise last_exception
    
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
    ) -> AsyncGenerator[ChatCompletionChunkChoice, None]:
        """Send streaming request.

        Yields parsed `ChatCompletionChunkChoice` objects for each chunk choice.
        This function only parses the SSE stream and does not perform tool execution.
        Tool handling is done in the caller (`chat_stream`) so it can control restarting streams.
        """
        url = f"{self.model_config.api_base_url}/chat/completions"
        headers = self._get_headers()
        request_data = request.to_dict(exclude_none=True)
        
        # Log HTTP request if enabled
        if self.runtime_config.capture_http_traffic:
            if self.runtime_config.log_http_requests:
                http_logger.info(f"POST {url} (streaming)")
                http_logger.debug(f"Request Headers:\n{json.dumps(dict(headers), indent=2)}")
                http_logger.debug(f"Request Body:\n{json.dumps(request_data, indent=2)}")
            
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
                
                # Log HTTP response headers if enabled (try to get info before consuming stream)
                if self.runtime_config.capture_http_traffic and self.runtime_config.log_http_responses:
                    try:
                        status_code = response.status_code
                        response_headers = dict(response.headers)
                        http_logger.info(f"HTTP/1.1 {status_code}")
                        http_logger.debug("Response Headers:")
                        for key, value in response_headers.items():
                            http_logger.debug(f"  {key}: {value}")
                        http_logger.debug("Response Body (SSE Stream):")
                    except Exception as e:
                        logger.debug(f"Could not log response headers: {e}")
                        status_code = 200
                        response_headers = {}
                else:
                    status_code = 200
                    response_headers = {}
                
                # Collect raw response lines for logging
                raw_response_lines = []
                
                line_count = 0
                async for line in response.aiter_lines():
                    line_count += 1
                    
                    # Log each raw SSE line (preserve original format)
                    if self.runtime_config.capture_http_traffic and self.runtime_config.log_http_responses:
                        # Log the original line exactly as received (SSE format)
                        http_logger.debug(f"  {line}")
                        raw_response_lines.append(line)
                    
                    line = line.strip()
                    
                    if not line or line.startswith(":"):
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]

                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)
                            
                            # Check if the chunk contains an error
                            if "error" in chunk_data:
                                error_info = chunk_data["error"]
                                error_msg = error_info.get("message", "Unknown error")
                                error_code = error_info.get("code", "unknown_error")
                                raise APIResponseError(
                                    f"API error in stream: {error_code} - {error_msg}",
                                    response_data=chunk_data
                                )
                            
                            chunk = ChatCompletionChunk.from_dict(chunk_data)

                            if chunk.choices and len(chunk.choices) > 0:
                                choice = chunk.choices[0]
                                # Yield the choice object to caller for higher-level handling
                                for _ in range(1):
                                    yield choice

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
                
                # Save raw response to debug file if enabled
                if self.runtime_config.capture_http_traffic and self.runtime_config.debug_save_responses:
                    self._save_debug_data("stream_response_raw", {
                        "status_code": status_code,
                        "headers": response_headers,
                        "raw_sse_lines": raw_response_lines
                    })
        
        except httpx.HTTPStatusError as e:
            # For streaming responses, we can't access .text directly
            try:
                error_text = e.response.text
            except Exception:
                error_text = f"HTTP {e.response.status_code}"
            
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

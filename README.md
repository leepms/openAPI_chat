# OpenAI-compatible Chat client (concise reference)

This package provides a compact, asyncio-based OpenAI-compatible chat client with three core features:

- Text chat (non-streaming and streaming)
- Multimodal inputs (image and video)
- Tool/function calling with optional automatic execution

Quick usage (minimal):

1) Install dependency:

```powershell
pip install httpx
```

2) Basic text chat example:

```python
import asyncio
from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig

async def main():
    cfg = ModelConfig(api_base_url="https://your.api", api_key="sk-...", model="gpt-3.5-turbo")
    rt = RuntimeConfig(enable_logging=True)
    async with ChatAgent(cfg, rt) as agent:
        resp = await agent.chat("Hello")
        print(resp)

asyncio.run(main())
```

3) Streaming example:

```python
# use `agent.chat_stream(...)` with `async for chunk in agent.chat_stream(...)` to receive chunks
```

4) Multimodal example:

```python
# pass image paths: image_paths="photo.jpg" or image_paths=["a.jpg","b.jpg"]
# pass video path: video_paths="video.mp4"
```

Examples (in `openai_chatapi/examples`):
- `example_1_basic_chat.py` — basic text chat
- `example_3_streaming.py` — streaming usage
- `example_4_multimodal.py` — image/video examples

That is all — this README contains only concise usage instructions and links to a few examples.
) -> str
```

**chat_stream()** - 流式对话
```python
async def chat_stream(
    text: str,
    image_paths: Union[str, List[str], None] = None,
    video_paths: Union[str, List[str], None] = None,
    add_to_history: bool = True,
    display_stream: bool = True,
    **kwargs
) -> AsyncGenerator[str, None]
```

**工具管理**
```python
register_tool(tool: Tool, handler: Callable) -> None
clear_tools() -> None
```

**消息管理**
```python
set_system_prompt(prompt: str) -> None
add_message(message: ChatMessage) -> None
clear_history(keep_system: bool = True) -> None
```

**统计信息**
```python
get_stats() -> dict
reset_stats() -> None
```

---

### ModelConfig

模型行为配置。

```python
@dataclass
class ModelConfig:
    # API连接
    api_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    
    # 采样参数
    temperature: float = 0.7
    top_p: float = 1.0
    
    # 长度控制
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    
    # 惩罚
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # 其他
    n: int = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
```

---

### RuntimeConfig

运行时行为配置。

```python
@dataclass
class RuntimeConfig:
    # 日志
    enable_logging: bool = True
    log_level: str = "INFO"
    log_http_requests: bool = False
    log_http_responses: bool = False
    
    # 监控
    enable_debug: bool = False
    capture_token_usage: bool = True
    capture_latency: bool = True
    
    # HTTP
    timeout: int = 60
    verify_ssl: bool = True
    max_retries: int = 0
    retry_delay: float = 1.0
    
    # 流式
    stream_chunk_callback: Optional[Callable[[str], None]] = None
    stream_enable_progress: bool = False
    
    # 解析
    strict_parsing: bool = False
    truncate_long_errors: bool = True
    max_error_length: int = 500
```

---

### 异常类型

所有异常继承自 `ChatAPIException`：

- **ConfigurationError** - 配置验证错误
- **APIConnectionError** - API 连接错误
- **APIResponseError** - API 响应解析错误
- **ToolExecutionError** - 工具执行错误
- **MediaProcessingError** - 媒体处理错误
- **ModelNotFoundError** - 模型未找到
- **TokenLimitError** - Token 限制超出

每个异常都包含 `message` 和 `details` 字典。

---

## 🎯 使用场景

### 场景1：生产环境部署

```python
# 完整的错误处理和监控
runtime_config = RuntimeConfig(
    enable_logging=True,
    log_level="WARNING",
    capture_token_usage=True,
    capture_latency=True,
    max_retries=3,
    timeout=30,
)

try:
    async with ChatAgent(model_config, runtime_config) as agent:
        response = await agent.chat(user_input)
        
        # 记录统计
        stats = agent.get_stats()
        logging.info(f"Request completed, tokens: {stats['total_tokens']}")
        
except ChatAPIException as e:
    logging.error(f"Chat failed: {e}")
    # 错误恢复逻辑
```

### 场景2：开发调试

```python
# 详细日志和调试信息
runtime_config = RuntimeConfig(
    enable_logging=True,
    log_level="DEBUG",
    log_http_requests=True,
    log_http_responses=True,
    enable_debug=True,
    verify_ssl=False,  # 本地测试
)

async with ChatAgent(model_config, runtime_config) as agent:
    # 所有请求/响应都会被详细记录
    response = await agent.chat("test")
```

### 场景3：流式前端展示

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat_endpoint(text: str):
    async def generate():
        async with ChatAgent(model_config) as agent:
            async for chunk in agent.chat_stream(text, display_stream=False):
                yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 📝 更新日志

### v0.3.0 (当前版本)

**新功能:**
- ✅ 视频输入支持
- ✅ 分离的配置系统（ModelConfig + RuntimeConfig）
- ✅ 完善的异常处理体系（8种异常类型）
- ✅ Token使用统计和延迟追踪
- ✅ 流式响应实时显示
- ✅ HTTP 请求/响应日志
- ✅ 调试模式

**改进:**
- ✅ 更好的错误诊断信息
- ✅ 自动截断长错误消息
- ✅ 向后兼容旧 API（ChatConfig）

**破坏性变更:**
- `ChatConfig` 改名为 `ModelConfig`（保留别名兼容）
- `ChatAgent.__init__` 现在接受两个配置参数

### v0.2.0

- 工具调用支持
- 模型管理模块
- 完整的 OpenAI 参数支持

### v0.1.0

- 基础文本对话
- 图像输入支持
- 流式响应

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License

---

## 🔗 相关链接

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [httpx 文档](https://www.python-httpx.org/)

---

## ❓ 常见问题

**Q: 如何使用本地模型服务（如 Ollama）？**

A: 只需修改 `api_base_url`：
```python
model_config = ModelConfig(
    api_base_url="http://localhost:11434/v1",
    api_key="not-needed",  # Ollama 不需要 key
    model="llama2",
)
```

**Q: 如何处理自签名 SSL 证书？**

A: 在 RuntimeConfig 中关闭验证：
```python
runtime_config = RuntimeConfig(verify_ssl=False)
```

**Q: 流式输出不显示怎么办？**

A: 确保启用了进度显示：
```python
runtime_config = RuntimeConfig(stream_enable_progress=True)
async for chunk in agent.chat_stream(text, display_stream=True):
    pass
```

**Q: 如何追踪 Token 使用量？**

A: 启用统计追踪：
```python
runtime_config = RuntimeConfig(capture_token_usage=True)
# ... 使用 agent ...
stats = agent.get_stats()
print(stats)
```

**Q: 支持哪些视频格式？**

A: 支持常见格式：mp4, webm, ogg, mov, avi。视频会被 Base64 编码后发送，注意大小限制。

**Q: 工具调用失败怎么办？**

A: 捕获 `ToolExecutionError` 查看详情：
```python
try:
    response = await agent.chat("...", auto_execute_tools=True)
except ToolExecutionError as e:
    print(f"Tool failed: {e.details['tool']}")
    print(f"Error: {e.details['error']}")
```

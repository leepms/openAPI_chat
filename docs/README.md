# OpenAI Compatible Chat API Client

一个功能完整的 OpenAI 兼容格式聊天 API 客户端，支持所有主流 OpenAI API 特性。

## ✨ 核心特性

### 🎯 多模态支持
- ✅ 文本、图像、**视频**输入（单个或多个）
- ✅ 灵活的内容组合方式
- ✅ 自动 Base64 编码处理

### 🔄 响应模式
- ✅ 非流式响应 - 完整响应一次返回
- ✅ 流式响应 - **实时显示**，支持进度回调
- ✅ 工具调用 - 自动执行和迭代控制

### 🛠️ 工具调用 (Function Calling)
- ✅ 完整的 OpenAI Tool/Function Calling 支持
- ✅ 同步和异步工具处理函数
- ✅ 自动工具执行和结果处理
- ✅ 多轮工具调用控制

### 📊 模型管理
- ✅ 自动获取可用模型列表
- ✅ 智能模型选择
- ✅ 模型详细信息查询

### ⚙️ 完整的 OpenAI API 参数
- ✅ 采样控制 - `temperature`, `top_p`
- ✅ 长度控制 - `max_tokens`, `max_completion_tokens`
- ✅ 惩罚参数 - `frequency_penalty`, `presence_penalty`
- ✅ 高级特性 - `logprobs`, `seed`, `stop`, `n`
- ✅ 推理模型 - `reasoning_effort` (o1 系列)

### 🔧 运行时配置
- ✅ **分离的配置系统** - 模型配置 + 运行时配置
- ✅ 日志控制 - 日志级别、HTTP 日志
- ✅ **统计追踪** - Token 使用量、延迟、成功率
- ✅ 调试模式 - 详细的调试输出
- ✅ SSL 控制 - 支持自签名证书

### 🛡️ 错误处理
- ✅ **完善的异常体系** - 8 种专用异常类型
- ✅ 详细的错误信息和诊断
- ✅ 自动错误跟踪和统计

### 💡 技术特点
- ✅ **最小依赖** - 仅需 httpx
- ✅ 类型安全 - 使用 Python dataclass
- ✅ 完全异步 - 基于 async/await
- ✅ 向后兼容 - 保持旧 API 兼容

---

## 📦 安装

```bash
pip install httpx
```

或使用 requirements.txt:

```bash
cd openai_chatapi
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 1. 基本文本对话

```python
import asyncio
from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig

async def main():
    # 模型配置
    model_config = ModelConfig(
        api_base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model="gpt-4o",
        temperature=0.7,
    )
    
    # 运行时配置（可选）
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
        capture_token_usage=True,
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        agent.set_system_prompt("You are a helpful assistant.")
        
        response = await agent.chat("Hello, how are you?")
        print(response)
        
        # 查看统计信息
        print(agent.get_stats())

asyncio.run(main())
```

**向后兼容写法**（使用 ChatConfig）:
```python
from openai_chatapi import ChatAgent, ChatConfig

config = ChatConfig(api_base_url="...", api_key="...", model="gpt-4o")
async with ChatAgent(config) as agent:
    response = await agent.chat("Hello!")
```

---

### 2. 多模态输入（图像+视频）

```python
# 图像输入
response = await agent.chat(
    "What's in this image?",
    image_paths="photo.jpg"
)

# 多张图像
response = await agent.chat(
    "Compare these images",
    image_paths=["image1.jpg", "image2.jpg"]
)

# 视频输入（新功能）
response = await agent.chat(
    "Describe what happens in this video",
    video_paths="video.mp4"
)

# 混合输入
response = await agent.chat(
    "Analyze these media files",
    image_paths=["photo1.jpg", "photo2.png"],
    video_paths="demo.mp4"
)
```

---

### 3. 流式响应（实时显示）

```python
# 启用实时显示
runtime_config = RuntimeConfig(
    stream_enable_progress=True,  # 实时打印
)

async with ChatAgent(model_config, runtime_config) as agent:
    # 流式输出会自动在终端显示
    async for chunk in agent.chat_stream(
        "Tell me a long story", 
        display_stream=True
    ):
        # chunk 会自动打印，也可以手动处理
        pass

# 自定义回调处理
def my_callback(chunk: str):
    # 处理每个chunk，例如发送到前端
    print(f"Received: {chunk}")

runtime_config = RuntimeConfig(
    stream_chunk_callback=my_callback
)
```

---

### 4. 工具调用（Function Calling）

```python
from openai_chatapi import Tool, FunctionDefinition

# 定义工具函数
def get_weather(location: str) -> dict:
    """获取天气信息"""
    return {"temperature": 22, "condition": "sunny", "location": location}

# 定义工具描述
weather_tool = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Get weather information for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    )
)

# 注册工具
agent.register_tool(weather_tool, get_weather)

# 自动执行工具调用
response = await agent.chat(
    "What's the weather in Paris?",
    auto_execute_tools=True,  # 自动执行
    max_tool_iterations=5      # 最多5轮
)
print(response)  # "The weather in Paris is sunny with 22°C."
```

---

### 5. 高级配置

```python
# 完整的模型配置
model_config = ModelConfig(
    api_base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    
    # 采样参数
    temperature=0.8,
    top_p=0.9,
    
    # 长度控制
    max_tokens=2000,
    
    # 惩罚参数
    frequency_penalty=0.5,
    presence_penalty=0.5,
    
    # 高级特性
    seed=42,              # 可复现性
    logprobs=True,        # 返回概率
    top_logprobs=3,       # Top-3 概率
    stop=["END"],         # 停止序列
    n=1,                  # 生成数量
)

# 完整的运行时配置
runtime_config = RuntimeConfig(
    # 日志配置
    enable_logging=True,
    log_level="DEBUG",
    log_http_requests=True,   # 记录HTTP请求
    log_http_responses=True,  # 记录HTTP响应
    
    # 监控和调试
    enable_debug=True,
    capture_token_usage=True,  # 追踪token使用
    capture_latency=True,      # 追踪延迟
    
    # HTTP行为
    timeout=120,
    verify_ssl=False,          # 支持自签名证书
    max_retries=3,             # 重试次数
    retry_delay=2.0,           # 重试延迟
    
    # 流式配置
    stream_enable_progress=True,        # 实时显示
    stream_chunk_callback=my_callback,  # 自定义回调
    
    # 错误处理
    strict_parsing=False,      # 宽松解析
    truncate_long_errors=True, # 截断长错误
)

async with ChatAgent(model_config, runtime_config) as agent:
    # 使用配置
    response = await agent.chat("Hello")
    
    # 获取统计信息
    stats = agent.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average latency: {stats['average_latency']}s")
    print(f"Success rate: {stats['success_rate']}%")
```

---

### 6. 模型管理

```python
from openai_chatapi import ModelManager

manager = ModelManager(
    api_base_url="https://api.openai.com/v1",
    api_key="your-key",
    verify_ssl=False  # 如需要
)

# 获取所有模型
models = await manager.list_models()
print(f"Available models: {models}")

# 获取详细信息
detailed = await manager.list_models_detailed()
for model in detailed:
    print(f"{model.id}: owned by {model.owned_by}")

# 自动选择最佳模型
selected = await manager.select_model(["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
print(f"Selected: {selected}")

# 获取特定模型信息
info = await manager.get_model_info("gpt-4o")
print(info.to_dict())
```

---

### 7. 推理模型 (o1 系列)

```python
# o1 系列模型特殊配置
model_config = ModelConfig(
    model="o1-preview",
    reasoning_effort="high",  # "low", "medium", "high"
    max_completion_tokens=5000,  # o1 使用 max_completion_tokens
)

async with ChatAgent(model_config) as agent:
    response = await agent.chat("Solve this complex math problem: ...")
    print(response)
```

---

### 8. 错误处理

```python
from openai_chatapi import (
    ChatAPIException,
    APIConnectionError,
    APIResponseError,
    ToolExecutionError,
    MediaProcessingError,
)

try:
    async with ChatAgent(model_config) as agent:
        response = await agent.chat("Hello", image_paths="invalid.jpg")

except MediaProcessingError as e:
    print(f"Media error: {e}")
    print(f"File: {e.details.get('file_path')}")
    print(f"Type: {e.details.get('media_type')}")

except APIConnectionError as e:
    print(f"Connection error: {e}")
    print(f"URL: {e.details.get('url')}")
    print(f"Status: {e.details.get('status_code')}")

except ToolExecutionError as e:
    print(f"Tool error: {e}")
    print(f"Tool: {e.details.get('tool')}")

except ChatAPIException as e:
    # 所有异常的基类
    print(f"API error: {e}")
    print(f"Details: {e.details}")
```

---

## 📚 API 参考

### ChatAgent

主要的聊天代理类。

#### 初始化
```python
ChatAgent(
    model_config: ModelConfig,
    runtime_config: RuntimeConfig = None
)
```

#### 核心方法

**chat()** - 非流式对话
```python
async def chat(
    text: str,
    image_paths: Union[str, List[str], None] = None,
    video_paths: Union[str, List[str], None] = None,
    add_to_history: bool = True,
    auto_execute_tools: bool = True,
    max_tool_iterations: int = 5,
    **kwargs
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

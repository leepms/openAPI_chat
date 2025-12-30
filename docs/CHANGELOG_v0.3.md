# OpenAI Chat API Client - v0.3.0 功能更新总结

## 🎉 更新概述

本次更新 (v0.3.0) 对模块进行了全面增强，重点改进了配置系统、错误处理、统计追踪和多模态支持。

---

## ✨ 主要新功能

### 1. ⚙️ 分离的配置系统

**之前：** 单一的 `ChatConfig` 混合了模型参数和客户端配置
```python
config = ChatConfig(
    model="gpt-4",
    temperature=0.7,
    timeout=60,      # 客户端配置
    verify_ssl=True, # 客户端配置
)
```

**现在：** 分离为 `ModelConfig` 和 `RuntimeConfig`
```python
# 模型行为配置
model_config = ModelConfig(
    model="gpt-4o",
    temperature=0.7,
    top_p=0.9,
)

# 运行时配置
runtime_config = RuntimeConfig(
    timeout=60,
    verify_ssl=False,
    enable_logging=True,
    capture_token_usage=True,
)

agent = ChatAgent(model_config, runtime_config)
```

**优势：**
- 🎯 清晰的责任分离
- 🔧 更灵活的配置管理
- 📊 增加了统计和监控配置
- ⚡ 向后兼容（ChatConfig 仍可用）

---

### 2. 🎬 视频输入支持

**新增功能：** 支持视频文件作为输入

```python
# 单个视频
response = await agent.chat(
    "描述这个视频内容",
    video_paths="demo.mp4"
)

# 多个视频
response = await agent.chat(
    "比较这些视频",
    video_paths=["video1.mp4", "video2.mp4"]
)

# 混合媒体（图像+视频）
response = await agent.chat(
    "分析这些媒体文件",
    image_paths=["photo.jpg"],
    video_paths=["video.mp4"]
)
```

**技术实现：**
- 自动 Base64 编码
- 支持多种视频格式（mp4, webm, ogg, mov, avi）
- 错误处理和异常提示

---

### 3. 📊 统计追踪系统

**新增 `UsageStats` 类：** 自动追踪 API 使用情况

```python
runtime_config = RuntimeConfig(
    capture_token_usage=True,
    capture_latency=True,
)

async with ChatAgent(model_config, runtime_config) as agent:
    # 使用 agent...
    
    # 获取统计
    stats = agent.get_stats()
    print(stats)
```

**追踪的指标：**
```python
{
    'total_requests': 10,
    'total_tokens': 1523,
    'prompt_tokens': 892,
    'completion_tokens': 631,
    'average_latency': 1.234,  # 秒
    'total_latency': 12.34,
    'errors': 0,
    'success_rate': 100.0,     # %
}
```

**使用场景：**
- 成本控制和预算管理
- 性能监控
- API 使用分析
- 错误率追踪

---

### 4. 🔔 流式实时显示

**增强的流式响应：** 支持实时终端显示和自定义回调

```python
# 方式1: 自动打印到终端
runtime_config = RuntimeConfig(
    stream_enable_progress=True
)

async for chunk in agent.chat_stream(text, display_stream=True):
    # chunk 会自动打印
    pass

# 方式2: 自定义回调处理
def handle_chunk(chunk: str):
    # 发送到前端、写入文件等
    print(f"Received: {chunk}")

runtime_config = RuntimeConfig(
    stream_chunk_callback=handle_chunk
)

async for chunk in agent.chat_stream(text):
    # 回调会自动处理每个chunk
    pass
```

**适用场景：**
- 终端应用实时显示
- Web 应用 SSE 推送
- 日志记录和监控
- 进度提示

---

### 5. 🛡️ 完善的异常处理

**新增 8 种专用异常类：**

```python
from openai_chatapi import (
    ChatAPIException,        # 基类
    ConfigurationError,      # 配置错误
    APIConnectionError,      # 连接错误
    APIResponseError,        # 响应解析错误
    ToolExecutionError,      # 工具执行错误
    MediaProcessingError,    # 媒体处理错误
    ModelNotFoundError,      # 模型未找到
    TokenLimitError,         # Token限制
)
```

**详细的错误信息：**

```python
try:
    response = await agent.chat("test", image_paths="invalid.jpg")
except MediaProcessingError as e:
    print(f"错误: {e.message}")
    print(f"文件: {e.details['file_path']}")
    print(f"类型: {e.details['media_type']}")
```

**优势：**
- 精确的错误定位
- 详细的诊断信息
- 更好的错误恢复
- 便于日志和监控

---

### 6. 📝 增强的日志系统

**新的日志配置选项：**

```python
runtime_config = RuntimeConfig(
    enable_logging=True,
    log_level="DEBUG",           # DEBUG, INFO, WARNING, ERROR
    log_http_requests=True,      # 记录HTTP请求内容
    log_http_responses=True,     # 记录HTTP响应内容
    enable_debug=True,           # 调试模式
)
```

**日志内容示例：**
```
2025-12-25 14:30:00 - openai_chatapi - INFO - POST https://api.openai.com/v1/chat/completions
2025-12-25 14:30:00 - openai_chatapi - DEBUG - Request: {"model": "gpt-4o", ...}
2025-12-25 14:30:01 - openai_chatapi - DEBUG - Response: {"id": "chatcmpl-...", ...}
2025-12-25 14:30:01 - openai_chatapi - DEBUG - Executing tool: get_weather with args: {"location": "Paris"}
```

---

## 🔧 技术改进

### 代码重构

1. **模块化设计**
   - `config.py` - 仅模型配置
   - `runtime_config.py` - 运行时配置和统计
   - `exceptions.py` - 异常定义
   - `chat_agent.py` - 核心逻辑增强

2. **更好的错误处理**
   - 所有 HTTP 错误都转换为自定义异常
   - 自动截断过长的错误消息
   - 详细的错误上下文信息

3. **性能追踪**
   - 自动记录请求延迟
   - Token 使用统计
   - 成功率计算

### 向后兼容性

**保持完全兼容：**

```python
# ✅ 旧代码仍然可用
from openai_chatapi import ChatAgent, ChatConfig

config = ChatConfig(...)  # ChatConfig = ModelConfig
agent = ChatAgent(config)  # 单参数仍支持

# ✅ 新代码使用新API
from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig

agent = ChatAgent(model_config, runtime_config)
```

**无需修改现有代码！**

---

## 📋 功能对比表

| 功能 | v0.2.0 | v0.3.0 |
|------|--------|--------|
| 文本对话 | ✅ | ✅ |
| 图像输入 | ✅ | ✅ |
| 视频输入 | ❌ | ✅ |
| 流式响应 | ✅ | ✅ 增强 |
| 实时显示 | ❌ | ✅ |
| 工具调用 | ✅ | ✅ |
| 模型管理 | ✅ | ✅ |
| 配置分离 | ❌ | ✅ |
| Token统计 | ❌ | ✅ |
| 延迟追踪 | ❌ | ✅ |
| 错误诊断 | 基础 | ✅ 详细 |
| 日志控制 | 基础 | ✅ 完善 |
| 自定义回调 | ❌ | ✅ |
| 调试模式 | ❌ | ✅ |

---

## 🚀 升级指南

### 最小改动升级（推荐）

```python
# 之前
from openai_chatapi import ChatAgent, ChatConfig

config = ChatConfig(...)
agent = ChatAgent(config)

# 现在 - 无需改动，完全兼容！
from openai_chatapi import ChatAgent, ChatConfig

config = ChatConfig(...)
agent = ChatAgent(config)
```

### 完整功能升级

```python
# 之前
config = ChatConfig(
    model="gpt-4o",
    temperature=0.7,
    timeout=60,
)
agent = ChatAgent(config)

# 现在 - 使用新配置系统
model_config = ModelConfig(
    model="gpt-4o",
    temperature=0.7,
)

runtime_config = RuntimeConfig(
    timeout=60,
    capture_token_usage=True,
    enable_logging=True,
)

agent = ChatAgent(model_config, runtime_config)
```

---

## 📚 新增文件

```
openai_chatapi/
├── exceptions.py         # 新增 - 异常定义
├── runtime_config.py     # 新增 - 运行时配置和统计
├── examples_v0.3.py      # 新增 - v0.3功能示例
├── README.md             # 更新 - 完整文档
├── config.py             # 重构 - 仅模型配置
├── chat_agent.py         # 增强 - 新功能集成
├── utils.py              # 扩展 - 视频支持
└── __init__.py           # 更新 - 导出新接口
```

---

## 🎯 使用建议

### 1. 开发环境

```python
runtime_config = RuntimeConfig(
    enable_logging=True,
    log_level="DEBUG",
    log_http_requests=True,
    enable_debug=True,
    verify_ssl=False,  # 本地测试
)
```

### 2. 生产环境

```python
runtime_config = RuntimeConfig(
    enable_logging=True,
    log_level="WARNING",  # 仅记录警告和错误
    capture_token_usage=True,
    capture_latency=True,
    max_retries=3,
    timeout=30,
)
```

### 3. 成本监控

```python
runtime_config = RuntimeConfig(
    capture_token_usage=True,
)

# 定期检查
stats = agent.get_stats()
if stats['total_tokens'] > 1000000:
    alert("Token usage high!")
```

### 4. 性能优化

```python
runtime_config = RuntimeConfig(
    capture_latency=True,
    stream_enable_progress=False,  # 生产环境关闭
    log_http_requests=False,       # 减少开销
)
```

---

## 📞 支持

如遇问题，请查看：
1. 完整文档：`README.md`
2. 功能示例：`examples_v0.3.py`
3. 旧示例（仍可用）：`examples_complete.py`

---

## 🎊 总结

v0.3.0 是一个重大更新，带来了：
- ✅ 更清晰的配置系统
- ✅ 更完善的错误处理
- ✅ 更强大的监控能力
- ✅ 更丰富的功能支持
- ✅ 完全向后兼容

**升级建议：** 立即升级，零风险！旧代码无需修改即可享受新功能。

**版本号：** 0.3.0
**发布日期：** 2025-12-25

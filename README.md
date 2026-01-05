# OpenAI-Compatible Chat API Client

一个轻量、功能完整的 OpenAI 兼容 API 客户端，基于 Python asyncio 实现。

## 📝 版本更新 (2026-01-05)

**v0.4.0 - 示例重构与文档优化**

### 主要变更
1. **示例代码重构** - 合并冗余示例，新增 4 个清晰的功能演示脚本
   - `example_1_basics.py` - 基础对话、多轮对话、多模态输入、回调函数
   - `example_2_streaming.py` - 流式输出的 4 种使用方式
   - `example_3_tool_calling.py` - 工具调用（非流式+流式）完整演示
   - `example_4_config_management.py` - 配置管理、HTTP 捕捉、Token 统计、调试模式

2. **文档清理** - 删除冗余文档，保留核心技术文档
   - 删除：`QUICKSTART.md`, `REFACTORING_REPORT.md`, `docs/CHANGELOG_v0.3.md` 等 7 个文件
   - 保留：`docs/CALLBACK_GUIDE.md`, `docs/STREAMING_TOOL_CALL_FIX.md` 等核心文档

3. **测试优化** - 修复导入路径，增加流式工具调用测试用例
   - 修复所有测试文件的模块导入（从相对导入改为绝对导入）
   - 新增 `test_stream_tool_execution()` 测试流式工具调用的参数拼接逻辑
---

## 🚀 核心特性

- ✅ **流式/非流式对话** - 支持实时响应输出
- 🎨 **多模态输入** - 图片、视频内容理解
- 🛠️ **工具调用** - Function Calling 自动执行
- 📊 **监控统计** - Token 使用、延迟、HTTP 捕捉
- ⚙️ **灵活配置** - YAML 文件 + 代码覆盖

## 📦 安装

```bash
pip install httpx pyyaml
```

## 🏗️ 项目结构

### 核心代码

```
openai_chatapi/
├── chat_agent.py        # ChatAgent 主类 - 对话管理、流式处理、工具调用
├── model_config.py      # ModelConfig - 模型参数配置 (temperature, max_tokens 等)
├── runtime_config.py    # RuntimeConfig - 运行时配置 (日志、监控、HTTP 捕捉)
│                        # UsageStats - Token 统计和延迟跟踪
├── schema.py            # 数据模型 - ChatMessage, Tool, ChatCompletionResponse 等
├── exceptions.py        # 异常类 - API 错误、工具执行错误等
├── model_manager.py     # 模型管理 - 模型列表、能力查询、推荐
├── utils/
│   ├── media_utils.py   # 媒体处理 - 图片/视频编码
│   └── config_loader.py # 配置加载 - YAML 解析
└── tools/
    ├── tool_loader.py   # 工具加载器
    └── fake_tool.py     # 示例工具
```

### 核心类说明

#### **ChatAgent** (`chat_agent.py`)
对话代理主类，提供完整的对话能力：
- `chat(text, image_paths, video_paths)` - 非流式对话
- `chat_stream(text, ...)` - 流式对话，yield 响应片段
- `register_tool(tool, handler)` - 注册工具函数
- `set_system_prompt(prompt)` - 设置系统提示词
- 自动处理工具调用：检测 → 执行 → 获取最终回复
- 支持多轮对话历史管理

#### **ModelConfig** (`model_config.py`)
模型行为配置：
- API 连接：`api_base_url`, `api_key`, `model`
- 采样参数：`temperature`, `top_p`, `frequency_penalty`
- 长度控制：`max_tokens`, `max_completion_tokens`
- 其他：`n`, `stop`, `seed`, `reasoning_effort`

#### **RuntimeConfig** (`runtime_config.py`)
运行时行为配置：
- **日志系统**：`log_level`, `save_logs_to_file`
- **HTTP 捕捉**：`capture_http_traffic`, `log_http_requests/responses`
- **统计监控**：`capture_token_usage`, `capture_latency`
- **调试模式**：`enable_debug`, `debug_save_requests/responses`
- **回调接口**：`response_callback`, `stream_chunk_callback`
- **终端控制**：`display_stream_output`

#### **UsageStats** (`runtime_config.py`)
统计信息跟踪：
- Token 使用量（prompt + completion）
- 请求延迟和平均值
- 成功率和错误计数

## 💡 快速开始

```python
import asyncio
from chat_agent import ChatAgent
from model_config import ModelConfig
from runtime_config import RuntimeConfig

async def main():
    # 配置
    model_cfg = ModelConfig(
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="your-api-key",
        model="qwen-plus",
        temperature=0.7
    )
    
    runtime_cfg = RuntimeConfig(
        enable_logging=True,
        capture_token_usage=True
    )
    
    # 使用
    async with ChatAgent(model_cfg, runtime_cfg) as agent:
        # 非流式
        response = await agent.chat("你好")
        print(response)
        
        # 流式
        async for chunk in agent.chat_stream("讲个笑话"):
            print(chunk, end='', flush=True)

asyncio.run(main())
```

**更多用法示例：**
- 工具调用、多模态输入 → 见 `examples/example_1_basics.py` 和 `example_3_tool_calling.py`
- HTTP 捕捉、Token 统计 → 见 `examples/example_4_config_management.py`

## 📚 示例脚本

`examples/` 目录包含 4 个完整示例：

### **example_1_basics.py** - 基础用法
- 功能 1: 单轮对话 + response_callback
- 功能 2: 多轮对话 + 历史记忆
- 功能 3: 多模态（图片分析）
- 功能 4: 自定义回调收集器

### **example_2_streaming.py** - 流式输出
- 示例 1: 自动显示流式输出
- 示例 2: 手动处理每个 chunk
- 示例 3: 回调函数处理（含工具调用说明）
- 示例 4: 禁用终端显示

### **example_3_tool_calling.py** - 工具调用
- 示例 1-2: 非流式工具调用（单工具 + 多工具）
- 示例 3: 流式工具调用
- 示例 4: 流式 + 禁用终端显示
- 示例 5: 自定义 chunk 处理器

### **example_4_config_management.py** - 配置与监控
- 示例 1: YAML 配置文件加载
- 示例 2: HTTP 报文捕捉（请求/响应详情）
- 示例 3: Token 使用统计（数量、延迟）
- 示例 4: 调试模式（保存 JSON 到文件）
- 示例 5: 回调函数配置
- 示例 6: 混合配置（YAML + 代码覆盖）

## 📖 配置参考

### ModelConfig 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_base_url` | str | openai.com/v1 | API 端点 |
| `api_key` | str | None | API 密钥 |
| `model` | str | "gpt-4o" | 模型名称 |
| `temperature` | float | 0.7 | 随机性 (0-2) |
| `max_tokens` | int | None | 最大输出长度 |
| `top_p` | float | 1.0 | 核采样 |
| `frequency_penalty` | float | 0.0 | 频率惩罚 |

### RuntimeConfig 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_level` | str | "INFO" | 日志级别 |
| `capture_http_traffic` | bool | False | HTTP 捕捉开关 |
| `capture_token_usage` | bool | True | Token 统计 |
| `capture_latency` | bool | True | 延迟统计 |
| `enable_debug` | bool | False | 调试模式 |
| `display_stream_output` | bool | True | 终端流式输出 |
| `response_callback` | Callable | None | 完整响应回调 |
| `stream_chunk_callback` | Callable | None | 流式 chunk 回调 |
| `max_tool_iterations` | int | 5 | 最大工具迭代次数 |

**完整参数说明：**参见示例脚本 `examples/example_4_config_management.py`

## 🧪 测试

```bash
# 运行所有测试
python test/test_chat_agent.py

# 测试真实 API
python test/test_real_api.py
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

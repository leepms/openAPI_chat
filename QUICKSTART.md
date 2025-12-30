# OpenAI Chat API Client - 快速参考

## 安装

```bash
pip install httpx pyyaml
```

## 基础使用

### 1. 使用YAML配置（推荐）

```python
from openai_chatapi import ChatAgent
from utils.config_loader import load_config_from_yaml

# 加载配置（支持参数覆盖）
model_config, runtime_config = load_config_from_yaml(
    api_key="your-key",
    temperature=0.7
)

# 使用
async with ChatAgent(model_config, runtime_config) as agent:
    response = await agent.chat("Hello!")
    print(response)
```

### 2. 命令行运行

```bash
# 使用默认配置
python examples/run_with_config.py --api-key your-key

# 覆盖参数
python examples/run_with_config.py \
  --temperature 0.9 \
  --log-level DEBUG \
  --prompt "你好"
```

### 3. 代码配置（传统方式）

```python
from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig

model_config = ModelConfig(
    api_key="your-key",
    model="gpt-4o",
    temperature=0.7
)
runtime_config = RuntimeConfig(log_level="INFO")

async with ChatAgent(model_config, runtime_config) as agent:
    response = await agent.chat("Hello!")
```

## 常用功能

### 多模态输入

```python
# 图像
response = await agent.chat("分析图像", image_paths="photo.jpg")

# 视频
response = await agent.chat("描述视频", video_paths="video.mp4")

# 混合
response = await agent.chat(
    "分析这些文件",
    image_paths=["img1.jpg", "img2.jpg"],
    video_paths="video.mp4"
)
```

### 流式输出

```python
async for chunk in agent.chat_stream("讲个故事", display_stream=True):
    pass  # 自动实时显示
```

### 工具调用

```python
from openai_chatapi import Tool, FunctionDefinition

def get_weather(location: str) -> dict:
    return {"temp": 22, "location": location}

tool = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}}
    )
)

agent.register_tool(tool, get_weather)
response = await agent.chat("What's the weather in Paris?", auto_execute_tools=True)
```

### 统计追踪

```python
stats = agent.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
agent.save_stats("logs/stats.log")
```

## 配置示例

### 开发环境

```python
model_config, runtime_config = load_config_from_yaml(
    log_level="DEBUG",
    enable_debug=True
)
```

### 生产环境

```python
model_config, runtime_config = load_config_from_yaml(
    save_logs_to_file=True,
    capture_token_usage=True,
    max_retries=3
)
```

## 错误处理

```python
from openai_chatapi import APIConnectionError, MediaProcessingError

try:
    response = await agent.chat("test", image_paths="photo.jpg")
except MediaProcessingError as e:
    print(f"Error: {e.message}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
```

## 版本信息

- **当前版本**: v0.3.1
- **Python要求**: >=3.8
- **依赖**: httpx, pyyaml

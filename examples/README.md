# OpenAI ChatAPI 示例代码

本目录包含各种使用场景的示例代码，帮助您快速上手。

## 📚 示例列表

### 1. 基础对话 (`example_1_basic_chat.py`)
最简单的单次对话示例。
```bash
python example_1_basic_chat.py
```
**学习要点:**
- 创建 ChatAgent
- 配置 ModelConfig 和 RuntimeConfig
- 发送消息并获取响应
- 查看统计信息

---

### 2. 多轮对话 (`example_2_multi_turn.py`)
演示如何进行连续对话并保持上下文。
```bash
python example_2_multi_turn.py
```
**学习要点:**
- 设置系统提示词
- 多轮对话保持上下文
- 查看对话历史
- 清除历史

---

### 3. 流式输出 (`example_3_streaming.py`)
实时显示 AI 响应（打字机效果）。
```bash
python example_3_streaming.py
```
**Examples (简洁)**

保留的示例：

- `example_1_basic_chat.py` — 基础对话示例（单轮/短提示）。
- `example_3_streaming.py` — 流式输出示例（实时 chunk 打印）。
- `example_4_multimodal.py` — 多模态示例（图片/视频输入）。

如何运行示例：

1. 进入示例目录：
   - `cd openai_chatapi/examples`
2. 设置 API key（示例会优先使用文件顶部的常量，若留空会回退到环境变量或 `config/default_model_config.yaml`）：
   - PowerShell: `$env:OPENAI_API_KEY = "your-key"`
3. 运行示例：
   - `python example_1_basic_chat.py`

测试：请使用项目根目录下的 `tests/` 测试套件，示例目录的历史测试已归档。

如需恢复其他被归档的示例，请告诉我需要恢复的文件名。

（此 README 已简化，仅保留最常用示例的运行说明。）
### 5. 工具调用 (`example_5_tool_calling.py`)

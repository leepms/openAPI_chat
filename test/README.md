# OpenAI ChatAPI 测试文档

本目录包含完整的测试套件，用于验证模块的所有功能。

## 📋 测试文件

### 1. `test_all_features.py` - 虚拟测试
**用途**: 测试模块结构和功能完整性，不需要真实 API

**特点**:
- ✅ 无需 API Key
- ✅ 快速执行（秒级）
- ✅ 测试所有功能的代码结构
- ✅ 生成详细测试报告

**运行**:
```bash
cd openai_chatapi/test
python test_all_features.py
```

**测试内容**:
1. ✅ 配置对象创建（ModelConfig, RuntimeConfig）
2. ✅ Agent 初始化和上下文管理
3. ✅ 消息管理（系统提示、添加、清除）
4. ✅ 工具管理（注册、清除）
5. ✅ 统计信息追踪
6. ✅ 方法签名验证
7. ✅ 错误类定义
8. ✅ Schema 数据结构
9. ✅ 请求构建逻辑

---

### 2. `test_real_api.py` - 真实 API 测试
**用途**: 与真实 API 接口交互，验证实际功能

**特点**:
- ⚠️ 需要有效 API Key
- ⚠️ 会产生 API 调用费用
- ✅ 测试完整功能流程
- ✅ 可配置跳过昂贵测试

**运行**:
```bash
# 方式 1: 使用环境变量（推荐）
export OPENAI_API_KEY="your-api-key"
python test_real_api.py

# 方式 2: 命令行参数
python test_real_api.py --api-key "your-key"

# 方式 3: 自定义 API 地址
python test_real_api.py --base-url "https://your-api.com/v1"

# 包含昂贵测试（如多模态）
python test_real_api.py --include-expensive

# 使用特定模型
python test_real_api.py --model gpt-4o
```

**测试内容**:
1. 🌐 基础 API 连接
2. 💬 多轮对话与上下文保持
3. 🌊 流式输出
4. 🔧 工具调用（Function Calling）
5. 🖼️ 多模态输入（图片）*
6. ❌ 错误处理和重试
7. ⚙️ 参数动态覆盖
8. 📊 统计追踪

*多模态测试需要 `--include-expensive` 参数

---

## 🚀 快速开始

### 步骤 1: 虚拟测试（无需 API）
```bash
cd openai_chatapi/test
python test_all_features.py
```

预期结果：
```
======================================================================
测试总结
======================================================================
总测试数: 20
✅ 通过: 18
❌ 失败: 0
⏭️  跳过: 2
通过率: 90.0%
======================================================================
```

---

### 步骤 2: 真实 API 测试
```bash
# 设置 API Key
export OPENAI_API_KEY="sk-your-key-here"

# 运行测试
python test_real_api.py
```

预期结果：
```
======================================================================
测试总结
======================================================================
总测试数: 8
✅ 通过: 7
❌ 失败: 0
⏭️  跳过: 1
🎉 所有测试通过！
======================================================================
```

---

## 📊 测试报告

测试完成后会生成 JSON 格式的报告：

- `test_report.json` - 虚拟测试报告
- `test_report_real_api.json` - 真实 API 测试报告

报告示例：
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "mode": "mock",
  "summary": {
    "total": 20,
    "passed": 18,
    "failed": 0,
    "skipped": 2
  },
  "tests": [
    {
      "name": "ModelConfig 创建",
      "status": "PASS",
      "message": "",
      "timestamp": "2025-01-01T12:00:01"
    }
  ]
}
```

---

## 🔧 高级用法

### 自定义测试配置

编辑 `test_real_api.py` 中的 `TestConfig` 类：

```python
class TestConfig:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.test_model = "gpt-4o-mini"  # 修改测试模型
        self.test_timeout = 30.0          # 修改超时时间
```

### 添加自定义测试

在 `test_real_api.py` 中添加新的测试函数：

```python
async def test_9_your_custom_test(test_config: TestConfig):
    """测试 9: 你的自定义测试"""
    print("\n" + "=" * 70)
    print("测试 9: 自定义测试描述")
    print("=" * 70)
    
    try:
        # 你的测试逻辑
        async with ChatAgent(...) as agent:
            # ...
            pass
        
        print("\n✅ 自定义测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 自定义测试失败: {e}")
        return False
```

然后在 `run_real_api_tests()` 中调用：
```python
results['your_test'] = await test_9_your_custom_test(test_config)
```

---

## ⚠️ 注意事项

### API 费用
真实 API 测试会产生费用：
- 使用 `gpt-4o-mini` 成本约 **$0.001-0.002** 每次完整测试
- 使用 `gpt-4o` 成本约 **$0.01-0.02** 每次完整测试
- 多模态测试会额外增加成本

### 网络要求
- 需要能够访问 OpenAI API
- 如果使用代理，请配置环境变量：
  ```bash
  export HTTP_PROXY="http://your-proxy:port"
  export HTTPS_PROXY="http://your-proxy:port"
  ```

### 测试数据
多模态测试需要测试图片：
```bash
# 放置测试图片
cp your_test_image.jpg openai_chatapi/test/test_image.jpg
```

---

## 🐛 故障排查

### 问题 1: ModuleNotFoundError
**解决**: 确保在项目根目录运行，或检查 Python 路径
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 问题 2: API Key 错误
**解决**: 检查 API key 是否正确设置
```bash
echo $OPENAI_API_KEY  # 应该显示你的 key
```

### 问题 3: 连接超时
**解决**: 增加超时时间或检查网络
```python
runtime_config = RuntimeConfig(timeout=60.0)
```

### 问题 4: 工具调用失败
**解决**: 确保 `tools/fake_tool.json` 存在且格式正确
```bash
ls -la openai_chatapi/tools/fake_tool.json
```

---

## 📈 持续集成

### GitHub Actions 示例

```yaml
name: API Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run mock tests
        run: |
          cd openai_chatapi/test
          python test_all_features.py
      - name: Run real API tests
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd openai_chatapi/test
          python test_real_api.py
```

---

## 📝 测试检查清单

在提交代码前，确保：

- [ ] ✅ 虚拟测试全部通过
- [ ] ✅ 真实 API 基础测试通过
- [ ] ✅ 没有引入新的异常或错误
- [ ] ✅ 代码符合项目规范
- [ ] ✅ 更新了相关文档
- [ ] ✅ 添加了必要的测试用例

---

## 📚 相关文档

- [主文档](../README.md)
- [示例代码](../examples/README.md)
- [配置说明](../config/)
- [工具系统](../tools/README.md)

---

**最后更新**: 2024-12-25  
**版本**: v0.3.1

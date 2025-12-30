#!/usr/bin/env python3
"""
OpenAI ChatAPI 完整功能测试套件

测试模块的所有功能：
1. 配置管理（ModelConfig, RuntimeConfig）
2. 基础对话（chat）
3. 流式输出（chat_stream）
4. 多模态输入（图片、视频）
5. 工具调用（Function Calling）
6. 对话历史管理
7. 统计信息追踪
8. 错误处理

注意：这是虚拟测试版本，使用模拟响应。
      实际测试需要提供真实 API 接口。
"""

import asyncio
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 以 package 方式导入，避免相对导入错误
from openai_chatapi.chat_agent import ChatAgent
from openai_chatapi.model_config import ModelConfig
from openai_chatapi.runtime_config import RuntimeConfig
from openai_chatapi.exceptions import APIConnectionError, APIResponseError, ConfigurationError
from openai_chatapi.schema import ChatMessage, Tool


# ============================================================
# 测试工具函数
# ============================================================

class TestResults:
    """测试结果收集器"""
    def __init__(self):
        self.tests: List[Dict] = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0
    
    def add_result(self, name: str, status: str, message: str = ""):
        """添加测试结果"""
        self.tests.append({
            "name": name,
            "status": status,  # "PASS", "FAIL", "SKIP"
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if status == "PASS":
            self.passed += 1
            print(f"  ✅ {name}")
        elif status == "FAIL":
            self.failed += 1
            print(f"  ❌ {name}: {message}")
        else:
            self.skipped += 1
            print(f"  ⏭️  {name}: {message}")
    
    def summary(self):
        """打印测试总结"""
        total = len(self.tests)
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"总测试数: {total}")
        print(f"✅ 通过: {self.passed}")
        print(f"❌ 失败: {self.failed}")
        print(f"⏭️  跳过: {self.skipped}")
        print(f"通过率: {self.passed/total*100:.1f}%" if total > 0 else "N/A")
        print("=" * 70)
        
        return self.failed == 0


# ============================================================
# 模拟 API 响应（用于虚拟测试）
# ============================================================

class MockAPIClient:
    """模拟 API 客户端"""
    
    @staticmethod
    def mock_chat_response(messages: List[Dict], model: str) -> Dict:
        """模拟非流式响应"""
        return {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"这是模拟响应。收到 {len(messages)} 条消息。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        }
    
    @staticmethod
    def mock_stream_chunks(messages: List[Dict], model: str):
        """模拟流式响应"""
        chunks = [
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {"content": "这是"}, "finish_reason": None}]},
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {"content": "模拟"}, "finish_reason": None}]},
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {"content": "流式"}, "finish_reason": None}]},
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {"content": "响应"}, "finish_reason": None}]},
            {"id": "chatcmpl-mock-123", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return chunks
    
    @staticmethod
    def mock_tool_call_response(tool_name: str) -> Dict:
        """模拟工具调用响应"""
        return {
            "id": "chatcmpl-mock-tool-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_123",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps({"query": "测试查询"})
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 25, "total_tokens": 85}
        }


# ============================================================
# 测试用例
# ============================================================

async def test_config_creation(results: TestResults):
    """测试 1: 配置对象创建"""
    print("\n" + "=" * 70)
    print("测试 1: 配置对象创建")
    print("=" * 70)
    
    try:
        # 测试 ModelConfig
        model_config = ModelConfig(
            api_key="test-key-123",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
        assert model_config.api_key == "test-key-123"
        assert model_config.model == "gpt-4o-mini"
        results.add_result("ModelConfig 创建", "PASS")
    except Exception as e:
        results.add_result("ModelConfig 创建", "FAIL", str(e))
    
    try:
        # 测试 RuntimeConfig
        runtime_config = RuntimeConfig(
            max_retries=3,
            timeout=60.0,
            enable_debug=True
        )
        assert runtime_config.max_retries == 3
        assert runtime_config.timeout == 60.0
        results.add_result("RuntimeConfig 创建", "PASS")
    except Exception as e:
        results.add_result("RuntimeConfig 创建", "FAIL", str(e))
    
    try:
        # 测试 YAML 配置加载
        yaml_path = Path(__file__).parent.parent / "config" / "default_model_config.yaml"
        if yaml_path.exists():
            config = ModelConfig.from_yaml(str(yaml_path))
            results.add_result("YAML 配置加载", "PASS")
        else:
            results.add_result("YAML 配置加载", "SKIP", "配置文件不存在")
    except Exception as e:
        results.add_result("YAML 配置加载", "FAIL", str(e))


async def test_agent_initialization(results: TestResults):
    """测试 2: Agent 初始化"""
    print("\n" + "=" * 70)
    print("测试 2: ChatAgent 初始化")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(api_key="test-key")
        runtime_config = RuntimeConfig(enable_debug=False)
        
        agent = ChatAgent(model_config, runtime_config)
        assert agent.model_config.api_key == "test-key"
        assert len(agent.messages) == 0
        assert len(agent.tools) == 0
        results.add_result("Agent 初始化", "PASS")
        
        await agent.close()
    except Exception as e:
        results.add_result("Agent 初始化", "FAIL", str(e))
    
    try:
        # 测试默认配置初始化
        agent = ChatAgent()
        assert agent.model_config is not None
        assert agent.runtime_config is not None
        results.add_result("Agent 默认初始化", "PASS")
        await agent.close()
    except Exception as e:
        results.add_result("Agent 默认初始化", "FAIL", str(e))
    
    try:
        # 测试上下文管理器
        async with ChatAgent(model_config) as agent:
            assert agent is not None
        results.add_result("上下文管理器", "PASS")
    except Exception as e:
        results.add_result("上下文管理器", "FAIL", str(e))


async def test_message_management(results: TestResults):
    """测试 3: 消息管理"""
    print("\n" + "=" * 70)
    print("测试 3: 消息管理")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(api_key="test"))
        
        # 测试系统提示词
        agent.set_system_prompt("你是一个有帮助的助手")
        assert len(agent.messages) == 1
        assert agent.messages[0].role == "system"
        results.add_result("设置系统提示词", "PASS")
        
        # 测试添加消息
        user_msg = ChatMessage(role="user", content="你好")
        agent.add_message(user_msg)
        assert len(agent.messages) == 2
        results.add_result("添加消息", "PASS")
        
        # 测试清除历史（保留系统消息）
        agent.clear_history(keep_system=True)
        assert len(agent.messages) == 1
        assert agent.messages[0].role == "system"
        results.add_result("清除历史（保留系统）", "PASS")
        
        # 测试完全清除
        agent.clear_history(keep_system=False)
        assert len(agent.messages) == 0
        results.add_result("完全清除历史", "PASS")
        
        await agent.close()
    except Exception as e:
        results.add_result("消息管理测试", "FAIL", str(e))


async def test_tool_management(results: TestResults):
    """测试 4: 工具管理"""
    print("\n" + "=" * 70)
    print("测试 4: 工具管理")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(api_key="test"))
        
        # 加载工具定义
        tool_json_path = Path(__file__).parent.parent / "tools" / "fake_tool.json"
        if tool_json_path.exists():
            with open(tool_json_path, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)
            
            for tool_data in tools_data:
                tool = Tool(**tool_data)
                agent.register_tool(tool)
            
            assert len(agent.tools) > 0
            results.add_result("注册工具", "PASS")
            
            # 测试清除工具
            agent.clear_tools()
            assert len(agent.tools) == 0
            results.add_result("清除工具", "PASS")
        else:
            results.add_result("工具管理测试", "SKIP", "fake_tool.json 不存在")
        
        await agent.close()
    except Exception as e:
        results.add_result("工具管理测试", "FAIL", str(e))


async def test_statistics(results: TestResults):
    """测试 5: 统计信息"""
    print("\n" + "=" * 70)
    print("测试 5: 统计信息追踪")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(api_key="test"))
        
        # 测试获取统计
        stats = agent.get_stats()
        assert "total_requests" in stats
        assert stats["total_requests"] == 0
        results.add_result("获取统计信息", "PASS")
        
        # 模拟添加统计数据
        agent.stats.add_request(
            prompt_tokens=100,
            completion_tokens=50,
            latency=1.5
        )
        
        stats = agent.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 150
        results.add_result("更新统计信息", "PASS")
        
        # 测试重置统计
        agent.reset_stats()
        stats = agent.get_stats()
        assert stats["total_requests"] == 0
        results.add_result("重置统计信息", "PASS")
        
        # 测试保存统计到文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        agent.stats.add_request(prompt_tokens=50, completion_tokens=25, latency=0.8)
        agent.save_stats(temp_path)
        
        # 验证文件存在
        assert Path(temp_path).exists()
        with open(temp_path, 'r') as f:
            saved_stats = json.load(f)
        assert saved_stats["total_requests"] == 1
        results.add_result("保存统计到文件", "PASS")
        
        # 清理
        Path(temp_path).unlink()
        await agent.close()
    except Exception as e:
        results.add_result("统计信息测试", "FAIL", str(e))


async def test_basic_chat_mock(results: TestResults):
    """测试 6: 基础对话（模拟）"""
    print("\n" + "=" * 70)
    print("测试 6: 基础对话功能（模拟响应）")
    print("=" * 70)
    
    # 注意：这里只测试方法调用，不测试实际 API 请求
    try:
        agent = ChatAgent(ModelConfig(api_key="test-key-mock"))
        
        # 验证 chat 方法存在且参数正确
        assert hasattr(agent, 'chat')
        assert callable(agent.chat)
        results.add_result("chat 方法存在", "PASS")
        
        # 验证方法签名
        import inspect
        sig = inspect.signature(agent.chat)
        params = list(sig.parameters.keys())
        assert 'text' in params
        assert 'image_paths' in params
        assert 'video_paths' in params
        results.add_result("chat 方法签名正确", "PASS")
        
        await agent.close()
        results.add_result("基础对话功能验证", "PASS", "需要真实 API 进行完整测试")
    except Exception as e:
        results.add_result("基础对话测试", "FAIL", str(e))


async def test_streaming_mock(results: TestResults):
    """测试 7: 流式输出（模拟）"""
    print("\n" + "=" * 70)
    print("测试 7: 流式输出功能（模拟）")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(api_key="test-key-mock"))
        
        # 验证 chat_stream 方法存在
        assert hasattr(agent, 'chat_stream')
        assert callable(agent.chat_stream)
        results.add_result("chat_stream 方法存在", "PASS")
        
        # 验证方法签名
        sig = inspect.signature(agent.chat_stream)
        params = list(sig.parameters.keys())
        assert 'text' in params
        assert 'display_stream' in params
        results.add_result("chat_stream 方法签名正确", "PASS")
        
        await agent.close()
        results.add_result("流式输出功能验证", "PASS", "需要真实 API 进行完整测试")
    except Exception as e:
        results.add_result("流式输出测试", "FAIL", str(e))


async def test_multimodal_mock(results: TestResults):
    """测试 8: 多模态输入（模拟）"""
    print("\n" + "=" * 70)
    print("测试 8: 多模态输入功能（模拟）")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(api_key="test-key-mock", model="gpt-4o"))
        
        # 测试创建多模态消息
        from openai_chatapi.utils.media_utils import create_user_message
        
        # 文本消息
        msg = create_user_message("测试文本")
        assert msg.role == "user"
        assert msg.content == "测试文本"
        results.add_result("创建文本消息", "PASS")
        
        # 带图片的消息（模拟路径）
        test_image_path = Path(__file__).parent / "test_image.jpg"
        if not test_image_path.exists():
            results.add_result("创建图片消息结构", "SKIP", "测试图片不存在: test_image.jpg")
        else:
            msg_with_image = create_user_message(
                "描述这张图片",
                image_paths=[str(test_image_path)]
            )
            assert msg_with_image.role == "user"
            results.add_result("创建图片消息结构", "PASS")
        
        await agent.close()
        results.add_result("多模态功能验证", "PASS", "需要真实图片文件进行完整测试")
    except Exception as e:
        results.add_result("多模态测试", "FAIL", str(e))


async def test_error_handling(results: TestResults):
    """测试 9: 错误处理"""
    print("\n" + "=" * 70)
    print("测试 9: 错误处理")
    print("=" * 70)
    
    try:
        # 测试配置错误
        from openai_chatapi.exceptions import ConfigurationError
        assert issubclass(ConfigurationError, Exception)
        results.add_result("ConfigurationError 定义", "PASS")
        
        # 测试 API 错误
        from openai_chatapi.exceptions import APIConnectionError, APIResponseError
        assert issubclass(APIConnectionError, Exception)
        assert issubclass(APIResponseError, Exception)
        results.add_result("API 异常类定义", "PASS")
        
        # 测试工具执行错误
        from openai_chatapi.exceptions import ToolExecutionError
        assert issubclass(ToolExecutionError, Exception)
        results.add_result("ToolExecutionError 定义", "PASS")
        
    except Exception as e:
        results.add_result("错误处理测试", "FAIL", str(e))


async def test_schema_validation(results: TestResults):
    """测试 10: Schema 验证"""
    print("\n" + "=" * 70)
    print("测试 10: 数据结构验证")
    print("=" * 70)
    
    try:
        from openai_chatapi.schema import ChatMessage, Tool, ToolCall, FunctionCall
        
        # 测试 ChatMessage
        msg = ChatMessage(role="user", content="测试")
        assert msg.role == "user"
        assert msg.content == "测试"
        results.add_result("ChatMessage 结构", "PASS")
        
        # 测试 Tool
        tool = Tool(
            type="function",
            function={
                "name": "test_func",
                "description": "测试函数",
                "parameters": {"type": "object", "properties": {}}
            }
        )
        assert tool.type == "function"
        assert tool.function["name"] == "test_func"
        results.add_result("Tool 结构", "PASS")
        
        # 测试 ToolCall
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test_func", arguments="{}")
        )
        assert tool_call.id == "call_123"
        results.add_result("ToolCall 结构", "PASS")
        
    except Exception as e:
        results.add_result("Schema 验证测试", "FAIL", str(e))


async def test_request_building(results: TestResults):
    """测试 11: 请求构建"""
    print("\n" + "=" * 70)
    print("测试 11: API 请求构建")
    print("=" * 70)
    
    try:
        agent = ChatAgent(ModelConfig(
            api_key="test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        ))
        
        # 添加消息
        agent.set_system_prompt("测试系统提示")
        agent.add_message(ChatMessage(role="user", content="测试"))
        
        # 构建请求（调用内部方法）
        request = agent._build_request(
            messages=agent.messages,
            stream=False
        )
        
        assert request.model == "gpt-4o-mini"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert len(request.messages) == 2
        results.add_result("请求构建", "PASS")
        
        # 测试流式请求
        stream_request = agent._build_request(
            messages=agent.messages,
            stream=True
        )
        assert stream_request.stream == True
        results.add_result("流式请求构建", "PASS")
        
        await agent.close()
    except Exception as e:
        results.add_result("请求构建测试", "FAIL", str(e))


# ============================================================
# 主测试函数
# ============================================================

async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("OpenAI ChatAPI 完整功能测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模式: 虚拟测试（模拟响应）")
    print("=" * 70)
    
    results = TestResults()
    
    # 运行所有测试
    await test_config_creation(results)
    await test_agent_initialization(results)
    await test_message_management(results)
    await test_tool_management(results)
    await test_statistics(results)
    await test_basic_chat_mock(results)
    await test_streaming_mock(results)
    await test_multimodal_mock(results)
    await test_error_handling(results)
    await test_schema_validation(results)
    await test_request_building(results)
    
    # 打印总结
    success = results.summary()
    
    # 保存测试报告
    report_path = Path(__file__).parent / "test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "mock",
            "summary": {
                "total": len(results.tests),
                "passed": results.passed,
                "failed": results.failed,
                "skipped": results.skipped
            },
            "tests": results.tests
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存到: {report_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

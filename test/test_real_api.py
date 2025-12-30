#!/usr/bin/env python3
"""
OpenAI ChatAPI 真实 API 测试脚本

此脚本用于测试与真实 API 接口的连接和功能。
运行前需要：
1. 设置环境变量 OPENAI_API_KEY
2. 确保 API 接口可访问
3. （可选）配置自定义 base_url

使用方法：
    # 使用环境变量
    export OPENAI_API_KEY="your-api-key"
    python test_real_api.py

    # 或在代码中设置 API key（不推荐）
    python test_real_api.py --api-key "your-key"
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 使用包导入以避免相对导入错误
from openai_chatapi.chat_agent import ChatAgent
from openai_chatapi.model_config import ModelConfig
from openai_chatapi.runtime_config import RuntimeConfig
from openai_chatapi.exceptions import APIConnectionError, APIResponseError


# ============================================================
# 测试配置
# ============================================================

class TestConfig:
    """测试配置"""
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.test_model = "qwen3-8b"  # 使用 qwen 系列模型作为默认
        self.test_timeout = 30.0


# ============================================================
# 测试用例
# ============================================================

async def test_1_basic_connection(test_config: TestConfig):
    """测试 1: 基础连接测试"""
    print("\n" + "=" * 70)
    print("测试 1: 基础 API 连接")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
                api_base_url=test_config.base_url,
            model=test_config.test_model,
            max_tokens=50
        )
        
        runtime_config = RuntimeConfig(
            timeout=test_config.test_timeout,
            enable_debug=True
        )
        
        async with ChatAgent(model_config, runtime_config) as agent:
            print("发送测试消息...")
            response = await agent.chat("你好，请回复'测试成功'")
            
            print(f"\n响应: {response}")
            print(f"\n统计信息:")
            stats = agent.get_stats()
            print(f"  - 请求次数: {stats['total_requests']}")
            print(f"  - Token 使用: {stats['total_tokens']}")
            print(f"  - 平均响应时间: {stats.get('average_latency', 0.0):.3f}s")
            
        print("\n✅ 基础连接测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 基础连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_multi_turn_conversation(test_config: TestConfig):
    """测试 2: 多轮对话"""
    print("\n" + "=" * 70)
    print("测试 2: 多轮对话与上下文保持")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
                api_base_url=test_config.base_url,
            model=test_config.test_model,
            max_tokens=100
        )
        
        async with ChatAgent(model_config) as agent:
            # 设置系统提示
            agent.set_system_prompt("你是一个数学助手")
            
            # 第一轮
            print("\n轮次 1:")
            response1 = await agent.chat("请记住：x = 5")
            print(f"用户: 请记住：x = 5")
            print(f"助手: {response1}")
            
            # 第二轮
            print("\n轮次 2:")
            response2 = await agent.chat("x 的值是多少？")
            print(f"用户: x 的值是多少？")
            print(f"助手: {response2}")
            
            # 验证上下文
            if "5" in response2:
                print("\n✅ 多轮对话测试通过（上下文保持正确）")
                return True
            else:
                print("\n⚠️  多轮对话测试警告（上下文可能未保持）")
                return False
                
    except Exception as e:
        print(f"\n❌ 多轮对话测试失败: {e}")
        return False


async def test_3_streaming_output(test_config: TestConfig):
    """测试 3: 流式输出"""
    print("\n" + "=" * 70)
    print("测试 3: 流式输出")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
                api_base_url=test_config.base_url,
            model=test_config.test_model,
            max_tokens=100
        )
        
        async with ChatAgent(model_config) as agent:
            print("\n开始流式输出:")
            print("-" * 70)
            
            # 手动处理每个 chunk（较为稳定）
            print("手动处理流，每个 chunk 将被打印：")
            print("响应: ", end="", flush=True)
            async for chunk in agent.chat_stream(
                "数到 5",
                display_stream=False
            ):
                # chunk 为字符串内容，直接打印
                print(chunk, end="", flush=True)
            print()
            
            print("\n✅ 流式输出测试通过")
            return True
            
    except Exception as e:
        print(f"\n❌ 流式输出测试失败: {e}")
        return False


async def test_4_tool_calling(test_config: TestConfig):
    """测试 4: 工具调用"""
    print("\n" + "=" * 70)
    print("测试 4: Function Calling (工具调用)")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
            api_base_url=test_config.base_url,
            model=test_config.test_model,
            max_tokens=200
        )

        runtime_config = RuntimeConfig()

        async with ChatAgent(model_config, runtime_config) as agent:
            # 加载工具
            tool_json_path = Path(__file__).parent.parent / "tools" / "fake_tool.json"
            if not tool_json_path.exists():
                print("⚠️  工具定义文件不存在，跳过测试")
                return None

            # 加载并注册工具
            with open(tool_json_path, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)

            from openai_chatapi.schema import Tool, FunctionDefinition
            from openai_chatapi.tools.fake_tool import get_tool_function
            for tool_data in tools_data:
                func = tool_data.get("function")
                if isinstance(func, dict):
                    func_obj = FunctionDefinition(**func)
                else:
                    func_obj = None
                tool = Tool(type=tool_data.get("type", "function"), function=func_obj)
                # Try to attach a local handler if available
                handler = get_tool_function(func_obj.name) if func_obj else None
                agent.register_tool(tool, handler)

            print(f"已注册 {len(agent.tools)} 个工具")

            # 测试工具调用
            print("\n发送需要工具的请求...")
            response = await agent.chat(
                "帮我搜索：Python 异步编程",
                auto_execute_tools=True,
                max_tool_iterations=3
            )

            print(f"\n最终响应: {response}")

            # 检查是否有工具调用
            tool_messages = [msg for msg in agent.messages if getattr(msg, "role", None) == "tool"]
            if tool_messages:
                print(f"\n✅ 工具调用测试通过（调用了 {len(tool_messages)} 次工具）")
                return True
            else:
                print("\n⚠️  工具调用测试警告（未检测到工具调用）")
                return False

    except Exception as e:
        print(f"\n❌ 工具调用测试失败: {e}")
        return False


async def test_5_multimodal_input(test_config: TestConfig):
    """测试 5: 多模态输入"""
    print("\n" + "=" * 70)
    print("测试 5: 多模态输入（图片）")
    print("=" * 70)
    
    try:
        # 注意：需要支持视觉的模型
        model_config = ModelConfig(
            api_key=test_config.api_key,
            api_base_url=test_config.base_url,
            model="qwen3-vl-8b-instruct",
            max_tokens=200
        )
        
        # 检查是否有测试图片
        test_image_path = Path(__file__).parent / "test_image.jpg"
        if not test_image_path.exists():
            print("⚠️  测试图片不存在，跳过测试")
            print(f"   请将测试图片放在: {test_image_path}")
            return None
        
        async with ChatAgent(model_config) as agent:
            print(f"使用图片: {test_image_path}")
            
            response = await agent.chat(
                "请描述这张图片",
                image_paths=[str(test_image_path)]
            )
            
            print(f"\n响应: {response}")
            print("\n✅ 多模态输入测试通过")
            return True
            
    except Exception as e:
        print(f"\n❌ 多模态输入测试失败: {e}")
        print("   注意：此功能需要支持视觉的模型（如 gpt-4o）")
        return False


async def test_6_error_handling(test_config: TestConfig):
    """测试 6: 错误处理"""
    print("\n" + "=" * 70)
    print("测试 6: 错误处理和重试机制")
    print("=" * 70)
    
    try:
        # 使用无效的 API key 测试错误处理
        model_config = ModelConfig(
            api_key="invalid-key-for-testing",
                api_base_url=test_config.base_url,
            model=test_config.test_model
        )
        
        runtime_config = RuntimeConfig(
            max_retries=2,
            timeout=10.0
        )
        
        async with ChatAgent(model_config, runtime_config) as agent:
            try:
                await agent.chat("测试消息")
                print("\n⚠️  错误处理测试警告（应该抛出异常但没有）")
                return False
            except (APIConnectionError, APIResponseError) as e:
                print(f"\n✅ 错误处理测试通过（正确捕获异常: {type(e).__name__}）")
                return True
                
    except Exception as e:
        print(f"\n❌ 错误处理测试失败: {e}")
        return False


async def test_7_parameter_override(test_config: TestConfig):
    """测试 7: 参数覆盖"""
    print("\n" + "=" * 70)
    print("测试 7: 动态参数覆盖")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
                api_base_url=test_config.base_url,
            model=test_config.test_model,
            temperature=0.5,
            max_tokens=50
        )
        
        async with ChatAgent(model_config) as agent:
            # 使用默认参数
            print("\n测试 1: 使用默认参数")
            response1 = await agent.chat("说一个数字")
            print(f"响应 1: {response1}")
            
            # 覆盖温度参数（更随机）
            print("\n测试 2: 覆盖参数 (temperature=1.5)")
            response2 = await agent.chat(
                "说一个数字",
                temperature=1.5,
                max_tokens=30
            )
            print(f"响应 2: {response2}")
            
            print("\n✅ 参数覆盖测试通过")
            return True
            
    except Exception as e:
        print(f"\n❌ 参数覆盖测试失败: {e}")
        return False


async def test_8_statistics_tracking(test_config: TestConfig):
    """测试 8: 统计追踪"""
    print("\n" + "=" * 70)
    print("测试 8: Token 使用和统计追踪")
    print("=" * 70)
    
    try:
        model_config = ModelConfig(
            api_key=test_config.api_key,
                api_base_url=test_config.base_url,
            model=test_config.test_model,
            max_tokens=50
        )
        
        async with ChatAgent(model_config) as agent:
            # 发送多个请求
            for i in range(3):
                await agent.chat(f"测试消息 {i+1}")
            
            # 获取统计
            stats = agent.get_stats()
            
            print("\n统计信息:")
            print(f"  总请求数: {stats['total_requests']}")
            print(f"  总 Token: {stats['total_tokens']}")
            print(f"  平均响应时间: {stats.get('average_latency', 0.0):.3f}s")
            estimated_cost = stats.get('estimated_cost', None)
            if estimated_cost is not None:
                print(f"  预估成本: ${estimated_cost:.6f}")
            else:
                print("  预估成本: (未计算)")
            
            if stats['total_requests'] == 3:
                print("\n✅ 统计追踪测试通过")
                return True
            else:
                print("\n⚠️  统计追踪测试警告（请求数不匹配）")
                return False
                
    except Exception as e:
        print(f"\n❌ 统计追踪测试失败: {e}")
        return False


# ============================================================
# 主测试函数
# ============================================================

async def run_real_api_tests(test_config: TestConfig, skip_expensive: bool = True):
    """运行真实 API 测试"""
    print("\n" + "=" * 70)
    print("OpenAI ChatAPI 真实 API 测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模式: 真实 API 调用")
    print(f"测试模型: {test_config.test_model}")
    if test_config.base_url:
        print(f"API 地址: {test_config.base_url}")
    print("=" * 70)
    
    results = {}
    
    # 必要测试（低成本）
    print("\n▶️  运行必要测试...")
    results['basic_connection'] = await test_1_basic_connection(test_config)
    results['multi_turn'] = await test_2_multi_turn_conversation(test_config)
    results['streaming'] = await test_3_streaming_output(test_config)
    results['tool_calling'] = await test_4_tool_calling(test_config)
    results['error_handling'] = await test_6_error_handling(test_config)
    results['parameter_override'] = await test_7_parameter_override(test_config)
    results['statistics'] = await test_8_statistics_tracking(test_config)
    
    # 可选测试（可能需要额外资源）
    if not skip_expensive:
        print("\n▶️  运行可选测试...")
        results['multimodal'] = await test_5_multimodal_input(test_config)
    else:
        print("\n⏭️  跳过昂贵测试（多模态）")
        print("   使用 --include-expensive 参数运行完整测试")
        results['multimodal'] = None
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    print(f"总测试数: {total}")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"⏭️  跳过: {skipped}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {failed} 个测试失败")
    
    print("=" * 70)
    
    # 保存测试报告
    report_path = Path(__file__).parent / "test_report_real_api.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "real_api",
            "model": test_config.test_model,
            "results": {k: str(v) for k, v in results.items()}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存到: {report_path}")
    
    return 0 if failed == 0 else 1


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OpenAI ChatAPI 真实 API 测试")
    parser.add_argument('--api-key', help='API Key（不推荐，建议使用环境变量）')
    parser.add_argument('--base-url', help='自定义 API 地址')
    parser.add_argument('--include-expensive', action='store_true', 
                       help='包含昂贵的测试（如多模态）')
    parser.add_argument('--model', default='qwen3-8b', help='测试使用的模型')
    
    args = parser.parse_args()
    
    # 获取 API key
    import os
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ 错误: 未提供 API Key")
        print("\n请通过以下方式之一提供 API Key:")
        print("  1. 环境变量: export OPENAI_API_KEY='your-key'")
        print("  2. 命令行参数: --api-key 'your-key'")
        return 1
    
    # 创建测试配置
    test_config = TestConfig(api_key=api_key, base_url=args.base_url)
    test_config.test_model = args.model

    # 如果命令行未传 base-url，则尝试从环境变量读取
    if not test_config.base_url:
        import os
        env_base = os.getenv('OPENAI_API_BASE_URL')
        if env_base:
            test_config.base_url = env_base
    
    # 运行测试
    try:
        exit_code = asyncio.run(run_real_api_tests(
            test_config, 
            skip_expensive=not args.include_expensive
        ))
        return exit_code
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        return 1
    except Exception as e:
        print(f"\n\n测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

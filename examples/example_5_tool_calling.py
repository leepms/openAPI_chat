"""
ARCHIVED: example_5_tool_calling.py

Tool-calling examples have been consolidated; keep the test harness
`test/test_all_features.py` for fake tool examples, and use the
archived `tool_calling_demo.py` if needed.
"""

print("example_5_tool_calling.py archived. Use tests/test_all_features.py for tooling examples.")

if __name__ == "__main__":
    print("This example has been archived.")
    
    
#!/usr/bin/env python3
"""
示例 5: 工具调用 (Function Calling)
演示如何让 AI 调用工具函数
"""

# 可编辑：优先在这里填写 API 参数，留空 (None) 则使用环境变量或配置文件
API_KEY = "sk-0253dd96205d4d83b0b792e08dfaec06"  # e.g. "sk-..." 或 None
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # e.g. "https://api.openai.com/v1" 或 None
MODEL = "qwen3-32b"
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig, load_tools_for_agent

# 填充 API 参数
if API_KEY is None:
    API_KEY = os.getenv("OPENAI_API_KEY")
if API_BASE_URL is None:
    API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")

if API_KEY is None or API_BASE_URL is None:
    try:
        cfg_path = Path(__file__).parent.parent / "config" / "default_model_config.yaml"
        if cfg_path.exists():
            cfg = ModelConfig.from_yaml(str(cfg_path))
            if API_KEY is None:
                API_KEY = cfg.api_key
            if API_BASE_URL is None:
                API_BASE_URL = cfg.api_base_url
    except Exception:
        pass


async def main():
    print("=" * 60)
    print("示例 5: 工具调用")
    print("=" * 60)
    print()

    # 配置
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )

    runtime_config = RuntimeConfig(
        enable_logging=True,
        capture_token_usage=True,
    )

    async with ChatAgent(model_config, runtime_config) as agent:

        # 加载工具
        tools_dir = Path(__file__).parent.parent / "tools"
        tool_count = load_tools_for_agent(agent, tools_dir / "fake_tool.json")
        print(f"📦 已加载 {tool_count} 个工具\n")
        print("-" * 60 + "\n")

        # 示例 1: 单个工具调用
        print("【示例 1: 天气查询】\n")
        print("💬 用户: 北京的天气怎么样？\n")

        response = await agent.chat(
            "北京的天气怎么样？",
            auto_execute_tools=True  # 自动执行工具调用
        )

        print(f"🤖 助手: {response}\n")
        print("-" * 60 + "\n")

        # 示例 2: 多个工具调用
        print("【示例 2: 组合使用多个工具】\n")
        print("💬 用户: 帮我搜索'人工智能'并计算 2023 + 2024\n")

        response = await agent.chat(
            "帮我搜索'人工智能'并计算 2023 + 2024",
            auto_execute_tools=True,
            max_tool_iterations=10  # 允许多轮工具调用
        )

        print(f"🤖 助手: {response}\n")
        print("-" * 60 + "\n")

        # 示例 3: 复杂任务
        print("【示例 3: 复杂多步骤任务】\n")
        print("💬 用户: 查询北京天气，如果温度超过25度就搜索'避暑景点'\n")

        response = await agent.chat(
            "查询北京天气，如果温度超过25度就搜索'避暑景点'",
            auto_execute_tools=True,
            max_tool_iterations=10
        )

        print(f"🤖 助手: {response}\n")

        # 显示对话历史（包含工具调用）
        print("-" * 60 + "\n")
        print("📋 对话历史:")
        for i, msg in enumerate(agent.messages[-6:], 1):  # 只显示最近6条
            if msg.role == "assistant" and getattr(msg, 'tool_calls', None):
                print(f"  {i}. [助手] 调用工具: {[tc.function.name for tc in msg.tool_calls]}")
            elif msg.role == "tool":
                print(f"  {i}. [工具] 返回结果")
            else:
                role = {"user": "用户", "assistant": "助手"}.get(msg.role, msg.role)
                content = msg.content if isinstance(msg.content, str) else "[内容]"
                print(f"  {i}. [{role}] {content[:50]}...")

        # 统计信息
        stats = agent.get_stats()
        print("\n" + "=" * 60)
        print(f"总请求: {stats['total_requests']} | "
              f"总 Token: {stats['total_tokens']} | "
              f"平均延迟: {stats.get('average_latency', 0.0):.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    try:
        # 提醒用户设置 API key
        if not os.getenv("OPENAI_API_KEY") and API_KEY is None:
            print("⚠️  Warning: OPENAI_API_KEY environment variable not set (examples may need it)")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 提示:")
        print("  - 工具是模拟的，会返回随机数据")
        print("  - auto_execute_tools=True 会自动执行工具并反馈结果")
        print("  - max_tool_iterations 控制最多几轮工具调用")
        print("  - 可在 tools/fake_tool.py 查看可用工具")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

#!/usr/bin/env python3
"""
示例 3: 工具调用 (Function Calling)
演示如何让 AI 调用工具函数，包括非流式和流式两种模式

功能特性:
- 非流式工具调用（chat）
- 流式工具调用（chat_stream）
- 自定义回调函数处理响应
- 控制终端输出显示
"""

# 可编辑：优先在这里填写 API 参数，留空 (None) 则使用环境变量或配置文件
API_KEY = None  # 在此填写你的 API Key
API_BASE_URL = None  # 在此填写你的 API Base URL
MODEL = "qwen-plus"  # 在此填写你的模型名称

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat_agent import ChatAgent
from model_config import ModelConfig
from runtime_config import RuntimeConfig
from tools.tool_loader import load_tools_for_agent

# 填充 API 参数
if API_KEY is None:
    API_KEY = os.getenv("OPENAI_API_KEY")
if API_BASE_URL is None:
    API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")

if API_KEY is None or API_BASE_URL is None or MODEL is None:
    try:
        cfg_path = Path(__file__).parent.parent / "config" / "default_model_config.yaml"
        if cfg_path.exists():
            cfg = ModelConfig.from_yaml(str(cfg_path))
            if API_KEY is None:
                API_KEY = cfg.api_key
            if API_BASE_URL is None:
                API_BASE_URL = cfg.api_base_url
            if MODEL == "qwen-plus":  # 如果是默认值，从配置加载
                MODEL = cfg.model
    except Exception:
        pass


# ========== 回调函数示例 ==========

def chunk_callback(chunk: str):
    """每个流式chunk的回调"""
    # 可以在这里处理每个chunk，例如保存到文件、发送到前端等
    pass  # 这里我们不做额外处理，让默认的终端输出工作


def response_callback(response: str):
    """完整响应的回调（非流式）"""
    # 可以在这里处理完整响应
    print(f"\n[回调] 收到完整响应，长度: {len(response)} 字符")


async def main():
    print("=" * 70)
    print("示例 3: 工具调用（非流式 + 流式）")
    print("=" * 70)
    print()

    # 配置
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )

    # RuntimeConfig 配置说明：
    # - max_tool_iterations: 最大工具调用迭代次数
    # - tool_failure_policy: 工具失败处理策略 ('inject_message', 'raise', 'retry_once')
    # - stream_chunk_callback: 流式响应每个chunk的回调函数
    # - response_callback: 非流式响应的完整响应回调函数
    # - display_stream_output: 是否在终端显示流式输出
    runtime_config = RuntimeConfig(
        enable_logging=True,
        capture_token_usage=True,
        max_tool_iterations=10,
        tool_failure_policy='inject_message',
        stream_chunk_callback=chunk_callback,  # 流式chunk回调
        response_callback=response_callback,    # 完整响应回调
        display_stream_output=True,            # 显示流式输出到终端
    )

    async with ChatAgent(model_config, runtime_config) as agent:

        # 加载工具
        tools_dir = Path(__file__).parent.parent / "tools"
        tool_count = load_tools_for_agent(agent, tools_dir / "fake_tool.json")
        print(f"📦 已加载 {tool_count} 个工具\n")
        print("-" * 70 + "\n")

        # ==================== 非流式工具调用 ====================
        
        print("【示例 1: 非流式工具调用 - 天气查询】\n")
        print("💬 用户: 北京的天气怎么样？\n")

        response = await agent.chat(
            "北京的天气怎么样？",
            auto_execute_tools=True
        )

        print(f"🤖 助手: {response}\n")
        print("-" * 70 + "\n")

        # ==================== 非流式多工具调用 ====================
        
        print("【示例 2: 非流式多工具调用 - 组合使用】\n")
        print("💬 用户: 帮我搜索'人工智能'并计算 2023 + 2024\n")

        response = await agent.chat(
            "帮我搜索'人工智能'并计算 2023 + 2024",
            auto_execute_tools=True,
            max_tool_iterations=5
        )

        print(f"🤖 助手: {response}\n")
        print("-" * 70 + "\n")

        # ==================== 流式工具调用 ====================
        
        print("【示例 3: 流式工具调用 - 天气查询】\n")
        print("💬 用户: 查询上海的天气\n")
        print("🤖 助手: ", end='', flush=True)

        full_response = ""
        async for chunk in agent.chat_stream(
            "查询上海的天气",
            auto_execute_tools=True,
            display_stream=True  # 显示流式输出
        ):
            full_response += chunk

        print()  # 换行
        print("-" * 70 + "\n")

        # ==================== 流式工具调用（禁用终端输出）====================
        
        print("【示例 4: 流式工具调用（禁用终端显示）】\n")
        print("💬 用户: 查询深圳的天气\n")
        
        # 临时关闭终端显示
        original_display = agent.runtime_config.display_stream_output
        agent.runtime_config.display_stream_output = False
        
        print("🤖 助手: [流式输出已关闭，仅通过回调处理]")
        
        chunks_received = []
        async for chunk in agent.chat_stream(
            "查询深圳的天气",
            auto_execute_tools=True,
            display_stream=False  # 不在终端显示
        ):
            chunks_received.append(chunk)
            # 这里可以将chunk发送到你的前端、保存到文件等
        
        # 恢复终端显示设置
        agent.runtime_config.display_stream_output = original_display
        
        full_response = "".join(chunks_received)
        print(f"\n✅ 通过回调收到 {len(chunks_received)} 个chunks，总长度: {len(full_response)} 字符")
        print(f"📝 响应内容: {full_response}\n")
        print("-" * 70 + "\n")

        # ==================== 自定义回调示例 ====================
        
        print("【示例 5: 使用自定义回调处理流式响应】\n")
        print("💬 用户: 搜索'机器学习'\n")
        
        # 定义自定义回调
        collected_chunks = []
        
        def custom_chunk_handler(chunk: str):
            """自定义chunk处理器"""
            collected_chunks.append(chunk)
            # 这里可以实现自定义逻辑：
            # - 实时发送到WebSocket
            # - 保存到数据库
            # - 更新UI进度条等
        
        # 临时替换回调
        original_callback = agent.runtime_config.stream_chunk_callback
        agent.runtime_config.stream_chunk_callback = custom_chunk_handler
        
        print("🤖 助手: ", end='', flush=True)
        
        async for chunk in agent.chat_stream(
            "搜索'机器学习'",
            auto_execute_tools=True,
            display_stream=True
        ):
            pass  # chunk已通过回调处理
        
        print()
        print(f"✅ 自定义回调收到 {len(collected_chunks)} 个chunks")
        
        # 恢复原回调
        agent.runtime_config.stream_chunk_callback = original_callback
        
        print("-" * 70 + "\n")

        # ==================== 显示对话历史 ====================
        
        print("📋 对话历史（最近10条）:")
        for i, msg in enumerate(agent.messages[-10:], 1):
            if msg.role == "assistant" and getattr(msg, 'tool_calls', None):
                print(f"  {i}. [助手] 调用工具: {[tc.function.name for tc in msg.tool_calls]}")
            elif msg.role == "tool":
                result = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"  {i}. [工具] {result[:60]}...")
            else:
                role = {"user": "用户", "assistant": "助手"}.get(msg.role, msg.role)
                content = msg.content if isinstance(msg.content, str) else "[内容]"
                print(f"  {i}. [{role}] {content[:50]}...")

        # ==================== 统计信息 ====================
        
        stats = agent.get_stats()
        print("\n" + "=" * 70)
        print(f"📊 统计信息:")
        print(f"   总请求: {stats['total_requests']}")
        print(f"   总Token: {stats['total_tokens']}")
        print(f"   平均延迟: {stats.get('average_latency', 0.0):.2f}s")
        print("=" * 70)

        # ==================== 使用说明 ====================
        
        print("\n💡 使用提示:")
        print("   1. 非流式调用: 使用 agent.chat()")
        print("   2. 流式调用: 使用 agent.chat_stream()")
        print("   3. 回调函数:")
        print("      - stream_chunk_callback: 处理每个流式chunk")
        print("      - response_callback: 处理完整响应（非流式）")
        print("   4. 控制终端输出:")
        print("      - display_stream_output=True/False (在runtime_config中)")
        print("      - display_stream=True/False (在chat_stream参数中)")
        print("   5. 工具相关:")
        print("      - auto_execute_tools=True: 自动执行工具")
        print("      - max_tool_iterations: 控制最多几轮工具调用")
        print("      - tool_failure_policy: 工具失败处理策略")


if __name__ == "__main__":
    try:
        # 提醒用户设置 API key
        if not os.getenv("OPENAI_API_KEY") and API_KEY is None:
            print("⚠️  Warning: API Key not configured")
            print("   Please set API_KEY in the file or OPENAI_API_KEY environment variable\n")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 已取消")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

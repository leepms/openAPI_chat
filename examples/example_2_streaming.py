#!/usr/bin/env python3
"""
示例 2: 流式输出
演示如何实时显示 AI 响应（打字机效果）
"""

API_KEY = None
API_BASE_URL = None
MODEL = "qwen-plus"

import asyncio
import os
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接从根目录导入模块
from chat_agent import ChatAgent
from model_config import ModelConfig
from runtime_config import RuntimeConfig

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
    print("=" * 70)
    print("示例 2: 流式输出")
    print("=" * 70)
    print()

    # 配置（使用顶部变量或环境/配置文件）
    # 配置
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )

    runtime_config = RuntimeConfig(
        enable_logging=True,
        display_stream_output=True,
    )

    async with ChatAgent(model_config, runtime_config) as agent:

        # ==================== 示例 1: 自动显示 ====================
        
        print("【示例 1: 自动显示流式输出】")
        print("💬 用户: 请用两句话解释什么是人工智能。\n")
        print("🤖 助手: ", end="", flush=True)
        
        async for _chunk in agent.chat_stream(
            "请用两句话解释什么是人工智能。",
            display_stream=True  # 自动打印每个 chunk
        ):
            pass  # chunk 已自动打印

        print("\n" + "-" * 70 + "\n")

        # ==================== 示例 2: 手动处理 ====================
        
        print("【示例 2: 手动处理每个 chunk】")
        print("💬 用户: 列举三个编程语言。\n")
        print("🤖 助手: ", end="", flush=True)

        chunks = []
        async for chunk in agent.chat_stream(
            "列举三个编程语言。",
            display_stream=True  # 仍然显示
        ):
            chunks.append(chunk)
            # 可以在这里添加自定义逻辑

        full_response = "".join(chunks)
        print(f"\n📦 收到 {len(chunks)} 个 chunks，总长度 {len(full_response)} 字符")
        print("-" * 70 + "\n")

        # ==================== 示例 3: 使用回调函数 ====================
        # ==================== 示例 3: 使用回调函数 ====================
        
        print("【示例 3: 使用回调函数处理】\n")
        
        # 自定义回调收集器，可用于 WebSocket 推送、实时翻译、日志记录等
        class ChunkCollector:
            def __init__(self):
                self.chunks = []
                self.total_chars = 0
            
            def collect(self, chunk: str):
                """
                处理每个流式 chunk
                
                注意：agent 内部已处理流式工具调用逻辑：
                - 自动检测工具调用 (finish_reason: tool_calls)
                - 缓冲工具调用参数碎片
                - 执行工具并自动重启流式请求获取最终回复
                
                因此回调中只需处理文本内容，无需担心工具调用细节
                """
                self.chunks.append(chunk)
                self.total_chars += len(chunk)
                # 可在此添加：发送到 WebSocket、实时翻译、保存到文件等
        
        collector = ChunkCollector()
        agent.runtime_config.stream_chunk_callback = collector.collect
        
        print("💬 用户: 什么是机器学习？\n")
        print("🤖 助手: ", end="", flush=True)
        
        async for chunk in agent.chat_stream(
            "什么是机器学习？",
            display_stream=True
        ):
            pass
        
        print(f"\n✅ 回调收集了 {len(collector.chunks)} 个 chunks")
        print(f"📊 总字符数: {collector.total_chars}")
        
        agent.runtime_config.stream_chunk_callback = None
        print("-" * 70 + "\n")

        # ==================== 示例 4: 禁用终端显示 ====================
        
        print("【示例 4: 禁用终端显示（仅回调处理）】\n")
        
        agent.runtime_config.display_stream_output = False
        
        print("💬 用户: 数到 5\n")
        print("🤖 助手: [终端显示已关闭]")
        
        collected = []
        async for chunk in agent.chat_stream(
            "数到 5",
            display_stream=False
        ):
            collected.append(chunk)
        
        agent.runtime_config.display_stream_output = True
        
        full_text = "".join(collected)
        print(f"\n✅ 通过回调收到: {full_text}")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    try:
        if not os.getenv("OPENAI_API_KEY") and API_KEY is None:
            print("⚠️  Warning: API Key not configured\n")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 已取消")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

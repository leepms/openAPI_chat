#!/usr/bin/env python3
"""
示例 3: 流式输出
演示如何实时显示 AI 响应（打字机效果）
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

from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig

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
    print("示例 3: 流式输出")
    print("=" * 60)
    print()

    # 配置（使用顶部变量或环境/配置文件）
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )

    runtime_config = RuntimeConfig(
        enable_logging=True,
        stream_enable_progress=True,
    )

    async with ChatAgent(model_config, runtime_config) as agent:

        # 示例 1: 自动显示流式输出
        print("【示例 1: 自动显示】")
        print("💬 用户: 请用三句话介绍人工智能的发展历程。\n")
        print("🤖 助手: ", end="", flush=True)

        async for _chunk in agent.chat_stream(
            "请用三句话介绍人工智能的发展历程。",
            display_stream=True  # 自动打印每个 chunk
        ):
            pass  # chunk 已自动打印

        print("\n" + "-" * 60 + "\n")

        # 示例 2: 手动处理每个 chunk
        print("【示例 2: 手动处理】")
        print("💬 用户: 用一句话总结机器学习。\n")
        print("🤖 助手: ", end="", flush=True)

        chunks = []
        async for chunk in agent.chat_stream(
            "用一句话总结机器学习。",
            display_stream=False  # 不自动打印
        ):
            # 手动处理每个 chunk（可以添加自定义逻辑）
            print(chunk, end="", flush=True)
            chunks.append(chunk)

        full_response = "".join(chunks)
        print("\n")
        print(f"\n📝 完整响应长度: {len(full_response)} 字符")
        print(f"📦 流式 chunks: {len(chunks)} 个")

        # 统计信息
        stats = agent.get_stats()
        print("\n" + "=" * 60)
        print(f"请求次数: {stats['total_requests']} | "
              f"平均延迟: {stats.get('average_latency', 0.0):.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

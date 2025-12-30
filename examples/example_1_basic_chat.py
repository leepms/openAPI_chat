#!/usr/bin/env python3
"""
示例 1: 基础对话
演示最简单的单次对话用法
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

# 填充 API 参数：优先使用文件顶部的常量，再使用环境变量，最后尝试从配置文件加载
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
    print("示例 1: 基础对话")
    print("=" * 60)
    print()
    
    # 配置（使用顶端的 API 参数或环境/配置文件）
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
        temperature=0.7,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
    )
    
    # 创建 agent 并发送消息
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 用户: 你好，请简单介绍一下自己。\n")
        
        response = await agent.chat("你好，请简单介绍一下自己。")
        
        print(f"🤖 助手: {response}\n")
        
        # 显示统计信息
        stats = agent.get_stats()
        print("-" * 60)
        print(f"Token 使用: {stats['total_tokens']} | "
              f"延迟: {stats['average_latency']:.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

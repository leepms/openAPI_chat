#!/usr/bin/env python3
"""
示例 4: 配置管理与日志监控
演示 ModelConfig 和 RuntimeConfig 的各种配置方式，以及日志、HTTP 捕捉、统计等功能
"""

API_KEY = None
API_BASE_URL = None
MODEL = "qwen-plus"

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    print("示例 4: 配置管理与日志监控")
    print("=" * 70)
    print()

    config_dir = Path(__file__).parent.parent / "config"

    # ==================== 示例 1: YAML 配置文件 ====================
    
    print("【示例 1: YAML 配置文件】\n")
    
    try:
        model_config = ModelConfig.from_yaml(str(config_dir / "default_model_config.yaml"))
        runtime_config = RuntimeConfig.from_yaml(str(config_dir / "default_runtime_config.yaml"))
        
        print(f"✅ 已加载配置:")
        print(f"   模型: {model_config.model}")
        print(f"   温度: {model_config.temperature}")
        print(f"   日志级别: {runtime_config.log_level}\n")
        
    except Exception as e:
        print(f"❌ YAML 配置加载失败: {e}\n")
    
    print("-" * 70 + "\n")

    # ==================== 示例 2: HTTP 报文捕捉 ====================
    
    print("【示例 2: HTTP 报文捕捉】\n")
    
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
        capture_http_traffic=True,      # 主开关
        log_http_requests=True,         # 记录请求
        log_http_responses=True,        # 记录响应
        save_http_traffic_to_file=True, # 保存到文件
        http_traffic_file_path="logs/http_traffic.log",
    )
    
    print("✅ HTTP 捕捉配置:")
    print("   capture_http_traffic: True")
    print("   log_http_requests: True")
    print("   log_http_responses: True")
    print("   保存路径: logs/http_traffic.log\n")
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 发送请求: 你好\n")
        response = await agent.chat("你好", add_to_history=False)
        print(f"🤖 响应: {response[:30]}...\n")
        print("📝 HTTP 请求和响应已记录到日志\n")
    
    print("-" * 70 + "\n")

    # ==================== 示例 3: Token 使用统计 ====================
    
    print("【示例 3: Token 使用统计】\n")
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        capture_token_usage=True,
        save_token_usage_to_file=True,
        token_usage_file_path="logs/token_usage.log",
        capture_latency=True,
        save_latency_to_file=True,
        latency_file_path="logs/latency.log",
    )
    
    print("✅ 统计配置:")
    print("   capture_token_usage: True")
    print("   capture_latency: True")
    print("   保存到文件: True\n")
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 测试统计: 列举三个国家\n")
        response = await agent.chat("列举三个国家", add_to_history=False)
        print(f"🤖 响应: {response}\n")
        
        # 查看统计信息
        stats = agent.stats
        print("📊 统计信息:")
        print(f"   总请求数: {stats.total_requests}")
        print(f"   总 Token: {stats.total_tokens}")
        print(f"   提示 Token: {stats.prompt_tokens}")
        print(f"   完成 Token: {stats.completion_tokens}")
        print(f"   平均延迟: {stats.get_average_latency():.2f}s")
        print(f"   错误数: {stats.errors}\n")
    
    print("-" * 70 + "\n")

    # ==================== 示例 4: 调试模式 ====================
    
    print("【示例 4: 调试模式（保存请求/响应）】\n")
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="DEBUG",
        enable_debug=True,
        debug_save_requests=True,
        debug_save_responses=True,
        debug_output_dir="debug",
    )
    
    print("✅ 调试配置:")
    print("   enable_debug: True")
    print("   debug_save_requests: True")
    print("   debug_save_responses: True")
    print("   输出目录: debug/\n")
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 调试请求: 1+1等于几？\n")
        response = await agent.chat("1+1等于几？", add_to_history=False)
        print(f"🤖 响应: {response}\n")
        print("📁 请求和响应已保存到 debug/ 目录\n")
    
    print("-" * 70 + "\n")

    # ==================== 示例 5: 回调函数配置 ====================
    
    print("【示例 5: 回调函数配置】\n")
    
    class ResponseMonitor:
        """监控和分析响应"""
        def __init__(self):
            self.responses = []
            self.chunks = []
        
        def on_response(self, response: str):
            """处理完整响应"""
            self.responses.append(response)
            print(f"   [回调] 收到响应，长度: {len(response)} 字符")
        
        def on_chunk(self, chunk: str):
            """处理流式 chunk"""
            self.chunks.append(chunk)
    
    monitor = ResponseMonitor()
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        response_callback=monitor.on_response,
        stream_chunk_callback=monitor.on_chunk,
        display_stream_output=True,
    )
    
    print("✅ 回调配置:")
    print("   response_callback: monitor.on_response")
    print("   stream_chunk_callback: monitor.on_chunk\n")
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 非流式请求: 你好\n")
        response = await agent.chat("你好", add_to_history=False)
        print(f"🤖 响应: {response[:30]}...\n")
        
        print("💬 流式请求: 数到 3\n")
        print("🤖 流式: ", end="", flush=True)
        async for chunk in agent.chat_stream("数到 3", display_stream=True):
            pass
        print(f"\n   [回调] 收到 {len(monitor.chunks)} 个 chunks\n")
    
    print("-" * 70 + "\n")

    # ==================== 示例 6: 混合配置（YAML + 代码）====================
    
    print("【示例 6: 混合配置（推荐用于生产）】\n")
    
    try:
        # 从 YAML 加载基础配置
        model_config = ModelConfig.from_yaml(
            str(config_dir / "default_model_config.yaml")
        )
        runtime_config = RuntimeConfig.from_yaml(
            str(config_dir / "default_runtime_config.yaml")
        )
        
        # 通过代码覆盖关键参数
        model_config.temperature = 0.8
        runtime_config.log_level = "WARNING"
        runtime_config.display_stream_output = False
        runtime_config.capture_token_usage = True
        runtime_config.capture_http_traffic = False
        
        print("✅ 混合配置:")
        print(f"   基础: YAML 文件")
        print(f"   覆盖温度: {model_config.temperature}")
        print(f"   覆盖日志级别: {runtime_config.log_level}")
        print(f"   关闭 HTTP 捕捉（生产环境）")
        print(f"   关闭终端显示（生产环境）\n")
        
    except Exception as e:
        print(f"❌ 混合配置失败: {e}\n")
    
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

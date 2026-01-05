#!/usr/bin/env python3
"""
示例 1: 基础用法合集
演示最常用的功能：基础对话、多轮对话、多模态输入

包含功能：
1. 单次对话
2. 多轮对话（保持上下文）
3. 图片输入
4. 回调函数使用
"""

# 可编辑：优先在这里填写 API 参数，留空 (None) 则使用环境变量或配置文件
API_KEY = None  # 在此填写你的 API Key
API_BASE_URL = None  # 在此填写你的 API Base URL
MODEL = "qwen-plus"  # 基础对话模型
VISION_MODEL = "qwen-vl-plus"  # 多模态模型

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


# ========== 回调函数示例 ==========

def response_handler(response: str):
    """处理完整响应的回调函数"""
    # 这里可以实现：保存到数据库、日志记录、触发其他操作等
    print(f"\n[回调] 收到响应，长度: {len(response)} 字符")


async def main():
    print("=" * 70)
    print("示例 1: 基础用法合集")
    print("=" * 70)
    print()

    # ==================== 功能 1: 单次对话 ====================
    
    print("【功能 1: 单次对话】\n")
    
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
        temperature=0.7,
    )
    
    # 配置回调函数
    runtime_config = RuntimeConfig(
        enable_logging=True,
        response_callback=response_handler,  # 设置响应回调
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 用户: 你好，请用一句话介绍你自己。\n")
        
        response = await agent.chat("你好，请用一句话介绍你自己。")
        
        print(f"🤖 助手: {response}\n")
        print("-" * 70 + "\n")

    # ==================== 功能 2: 多轮对话 ====================
    
    print("【功能 2: 多轮对话（保持上下文）】\n")
    
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        # 设置系统提示词
        agent.set_system_prompt("你是一个友好的助手，记住用户告诉你的信息。")
        
        # 第一轮：介绍自己
        print("💬 用户: 我叫小明，我喜欢编程。\n")
        response = await agent.chat("我叫小明，我喜欢编程。")
        print(f"🤖 助手: {response}\n")
        
        # 第二轮：测试记忆
        print("💬 用户: 我叫什么名字？\n")
        response = await agent.chat("我叫什么名字？")
        print(f"🤖 助手: {response}\n")
        
        # 第三轮：继续上下文
        print("💬 用户: 我喜欢什么？\n")
        response = await agent.chat("我喜欢什么？")
        print(f"🤖 助手: {response}\n")
        
        # 查看对话历史
        print(f"📋 对话历史: {len(agent.messages)} 条消息")
        print("-" * 70 + "\n")

    # ==================== 功能 3: 多模态输入 ====================
    
    print("【功能 3: 多模态输入（图片分析）】\n")
    
    # 使用支持视觉的模型
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=VISION_MODEL,
        temperature=0.7,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        response_callback=response_handler,
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        
        # 示例：单张图片
        image_path = Path(__file__).parent.parent / "data" / "images" / "90cd85bc7c8223374e90e973e8711499_1766558927702_0001.jpg"
        
        if image_path.exists():
            print(f"💬 用户: [图片] 描述这张图片的内容。\n")
            
            try:
                response = await agent.chat(
                    "描述这张图片的内容。",
                    image_paths=str(image_path)
                )
                print(f"🤖 助手: {response}\n")
            except Exception as e:
                print(f"❌ 多模态调用失败: {e}\n")
        else:
            print(f"⚠️  图片文件不存在: {image_path}")
            print("💡 提示: 请将图片放在 data/images/ 目录下\n")
        
        print("-" * 70 + "\n")

    # ==================== 功能 4: 自定义回调 ====================
    
    print("【功能 4: 自定义回调函数】\n")
    
    # 自定义回调：统计和保存
    class ResponseCollector:
        def __init__(self):
            self.responses = []
        
        def collect(self, response: str):
            self.responses.append(response)
            word_count = len(response)
            print(f"[自定义回调] 收集响应 #{len(self.responses)}, {word_count} 字符")
    
    collector = ResponseCollector()
    
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        response_callback=collector.collect,  # 使用自定义回调
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        print("💬 用户: 什么是人工智能？\n")
        response = await agent.chat("什么是人工智能？")
        print(f"🤖 助手: {response}\n")
        
        print("💬 用户: 它有什么应用？\n")
        response = await agent.chat("它有什么应用？")
        print(f"🤖 助手: {response}\n")
        
        print(f"✅ 总共收集了 {len(collector.responses)} 个响应\n")

    # ==================== 使用说明 ====================
    
    print("=" * 70)
    print("💡 使用说明:")
    print()
    print("1️⃣  单次对话:")
    print("   response = await agent.chat('你好')")
    print()
    print("2️⃣  多轮对话:")
    print("   - 使用相同的 agent 实例")
    print("   - 自动保持上下文")
    print("   - 可以设置 system_prompt")
    print()
    print("3️⃣  多模态输入:")
    print("   response = await agent.chat('描述图片', image_paths='path/to/image.jpg')")
    print("   - 支持单张或多张图片")
    print("   - 需要使用支持视觉的模型")
    print()
    print("4️⃣  回调函数:")
    print("   runtime_config = RuntimeConfig(")
    print("       response_callback=your_function")
    print("   )")
    print("   - 用于处理完整响应")
    print("   - 可以保存到数据库、记录日志等")
    print()
    print("📚 更多示例:")
    print("   - example_3_streaming.py - 流式输出")
    print("   - example_5_tool_calling.py - 工具调用")
    print("   - example_6_config_management.py - 配置管理")
    print("=" * 70)


if __name__ == "__main__":
    try:
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

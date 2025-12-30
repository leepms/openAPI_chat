#!/usr/bin/env python3
"""
示例 4: 多模态对话
演示如何发送图片和视频进行对话
"""
# 可编辑：优先在这里填写 API 参数，留空 (None) 则使用环境变量或配置文件
API_KEY = "sk-0253dd96205d4d83b0b792e08dfaec06"  # e.g. "sk-..." 或 None
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # e.g. "https://api.openai.com/v1" 或 None
MODEL = "qwen3-vl-8b-instruct"
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
    print("示例 4: 多模态对话")
    print("=" * 60)
    print()
    
    # 配置（多模态需要支持视觉的模型）
    model_config = ModelConfig(
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
        temperature=0.7,
        max_tokens=500,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        
        # 示例 1: 单张图片
        print("【示例 1: 分析单张图片】\n")
        
        # 创建测试图片路径（实际使用时替换为真实图片路径）
        image_path = "..\\data\\images\\90cd85bc7c8223374e90e973e8711499_1766558927702_0001.jpg"
        
        print(f"💬 用户: [图片: {image_path}] 这张图片里有什么？\n")
        
        # 如果你已准备好图片文件，实际调用如下：
        try:
            response = await agent.chat(
                "这张图片里有什么？",
                image_paths=image_path
            )
            print(f"🤖 助手: {response}\n")
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"调用多模态接口失败: {e}\n")
        
        # 示例 2: 多张图片
        print("【示例 2: 对比多张图片】\n")
        
        image_paths = [
            "..\\data\\images\\90cd85bc7c8223374e90e973e8711499_1766558927702_0001.jpg",
            "..\\data\\images\\90cd85bc7c8223374e90e973e8711499_1766558927702_0002.jpg",
        ]
        
        print(f"💬 用户: [图片 x{len(image_paths)}] 比较这两张图片的区别。\n")
        
        try:
            response = await agent.chat(
                "比较这两张图片的区别。",
                image_paths=image_paths
            )
            print(f"🤖 助手: {response}\n")
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"调用多模态接口失败: {e}\n")
        
        # 示例 3: 视频输入
        print("【示例 3: 分析视频】\n")
        
        video_path = "..\\data\\videos\\v_00001754_0.mp4"
        
        print(f"💬 用户: [视频: {video_path}] 描述这个视频的内容。\n")
        
        try:
            response = await agent.chat(
                "描述这个视频的内容。",
                video_paths=video_path
            )
            print(f"🤖 助手: {response}\n")
        except Exception as e:
            print(f"调用多模态接口失败: {e}\n")
        
        # 模拟示例（不需要真实文件）
        print("⚠️  注意: 此示例需要真实的图片/视频文件")
        print("请取消代码中的注释并提供真实文件路径来运行\n")
        
        print("📝 使用说明:")
        print("  1. image_paths 可以是单个路径字符串或路径列表")
        print("  2. video_paths 支持视频文件路径")
        print("  3. 图片支持: .jpg, .jpeg, .png, .gif, .webp")
        print("  4. 系统会自动将文件编码为 base64")
        print("  5. 确保模型支持多模态功能（如 gpt-4o）")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

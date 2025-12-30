"""
ARCHIVED: example_6_config_management.py

Configuration management examples are archived. Use the project's
`config/` folder and `ModelConfig` utilities directly.
"""

print("example_6_config_management.py archived. See config/default_model_config.yaml for examples.")

async def example_yaml_config():
    """使用 YAML 配置文件"""
    
    print("【方式 1: YAML 配置文件】\n")
    
    config_dir = Path(__file__).parent.parent / "config"
    
    # 从 YAML 加载配置
    model_config = ModelConfig.from_yaml(config_dir / "default_model_config.yaml")
    runtime_config = RuntimeConfig.from_yaml(config_dir / "default_runtime_config.yaml")
    
    # 可以覆盖部分参数
    model_config.temperature = 0.8
    
    print(f"✅ 已加载配置:")
    print(f"   模型: {model_config.model}")
    print(f"   温度: {model_config.temperature}")
    print(f"   日志级别: {runtime_config.log_level}\n")
    
    return model_config, runtime_config

async def example_code_config():
    """在代码中直接配置"""
    
    print("【方式 2: 代码配置】\n")
    
    model_config = ModelConfig(
        api_key="your-api-key-here",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
        capture_token_usage=True,
    )
    
    print(f"✅ 已创建配置:")
    print(f"   模型: {model_config.model}")
    print(f"   温度: {model_config.temperature}\n")
    
    return model_config, runtime_config

async def example_env_config():
    """使用环境变量"""
    
    print("【方式 3: 环境变量】\n")
    
    import os
    
    # API key 可以从环境变量读取
    model_config = ModelConfig(
        model="gpt-4o-mini",
        # api_key 会自动从 OPENAI_API_KEY 环境变量读取
    )
    
    print(f"✅ 配置说明:")
    print(f"   API Key: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置（需要设置 OPENAI_API_KEY）'}")
    print(f"   模型: {model_config.model}\n")
    
    return model_config, RuntimeConfig()

async def example_hybrid_config():
    """混合配置方式"""
    
    print("【方式 4: 混合配置（推荐）】\n")
    
    config_dir = Path(__file__).parent.parent / "config"
    
    # 从 YAML 加载基础配置
    model_config = ModelConfig.from_yaml(
        config_dir / "default_model_config.yaml",
        # 通过参数覆盖
        temperature=0.9,
        max_tokens=1500,
    )
    
    runtime_config = RuntimeConfig.from_yaml(
        config_dir / "default_runtime_config.yaml",
        log_level="DEBUG",  # 覆盖日志级别
    )
    
    print(f"✅ 混合配置:")
    print(f"   基础来源: YAML 文件")
    print(f"   覆盖参数: temperature=0.9, max_tokens=1500")
    print(f"   最终温度: {model_config.temperature}\n")
    
    return model_config, runtime_config

API_KEY = "sk-0253dd96205d4d83b0b792e08dfaec06"  # e.g. "sk-..." 或 None
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # e.g. "https://api.openai.com/v1" 或 None
MODEL = "qwen3-32b"
async def main():
    print("=" * 60)
    print("示例 6: 配置管理")
    print("=" * 60)
    print()
    
    # 演示不同的配置方式
    configs = []
    
    try:
        configs.append(await example_yaml_config())
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"YAML 配置示例失败: {e}\n")
    try:
        configs.append(await example_hybrid_config())
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"混合配置示例失败: {e}\n")
    
    # 使用其中一个配置进行实际对话
    if configs:
        model_config, runtime_config = configs[1]  # 使用代码配置
        
        print("【配置测试】\n")
        print("💬 用户: 你好\n")
        
        # 注意：需要有效的 API key 才能运行
        """
        async with ChatAgent(model_config, runtime_config) as agent:
            response = await agent.chat("你好")
            print(f"🤖 助手: {response}\n")
        """
        
        print("⚠️  配置已准备好，取消注释上方代码并设置 API key 即可运行\n")
    
    print("=" * 60)
    print("📝 配置最佳实践:")
    print("  1. 开发环境: 使用 YAML + 环境变量")
    print("  2. 生产环境: 使用环境变量 + 代码覆盖")
    print("  3. 测试环境: 直接代码配置")
    print("  4. 敏感信息: 始终使用环境变量")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig


async def example_yaml_config():
    """使用 YAML 配置文件"""
    
    print("【方式 1: YAML 配置文件】\n")
    
    config_dir = Path(__file__).parent.parent / "config"
    
    # 从 YAML 加载配置
    model_config = ModelConfig.from_yaml(config_dir / "default_model_config.yaml")
    runtime_config = RuntimeConfig.from_yaml(config_dir / "default_runtime_config.yaml")
    
    # 可以覆盖部分参数
    model_config.temperature = 0.8
    
    print(f"✅ 已加载配置:")
    print(f"   模型: {model_config.model}")
    print(f"   温度: {model_config.temperature}")
    print(f"   日志级别: {runtime_config.log_level}\n")
    
    return model_config, runtime_config


async def example_code_config():
    """在代码中直接配置"""
    
    print("【方式 2: 代码配置】\n")
    
    model_config = ModelConfig(
        api_key="your-api-key-here",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
        capture_token_usage=True,
    )
    
    print(f"✅ 已创建配置:")
    print(f"   模型: {model_config.model}")
    print(f"   温度: {model_config.temperature}\n")
    
    return model_config, runtime_config


async def example_env_config():
    """使用环境变量"""
    
    print("【方式 3: 环境变量】\n")
    
    import os
    
    # API key 可以从环境变量读取
    model_config = ModelConfig(
        model="gpt-4o-mini",
        # api_key 会自动从 OPENAI_API_KEY 环境变量读取
    )
    
    print(f"✅ 配置说明:")
    print(f"   API Key: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置（需要设置 OPENAI_API_KEY）'}")
    print(f"   模型: {model_config.model}\n")
    
    return model_config, RuntimeConfig()


async def example_hybrid_config():
    """混合配置方式"""
    
    print("【方式 4: 混合配置（推荐）】\n")
    
    config_dir = Path(__file__).parent.parent / "config"
    
    # 从 YAML 加载基础配置
    model_config = ModelConfig.from_yaml(
        config_dir / "default_model_config.yaml",
        # 通过参数覆盖
        temperature=0.9,
        max_tokens=1500,
    )
    
    runtime_config = RuntimeConfig.from_yaml(
        config_dir / "default_runtime_config.yaml",
        log_level="DEBUG",  # 覆盖日志级别
    )
    
    print(f"✅ 混合配置:")
    print(f"   基础来源: YAML 文件")
    print(f"   覆盖参数: temperature=0.9, max_tokens=1500")
    print(f"   最终温度: {model_config.temperature}\n")
    
    return model_config, runtime_config


# 可编辑：优先在这里填写 API 参数，留空 (None) 则使用环境变量或配置文件
API_KEY = "sk-0253dd96205d4d83b0b792e08dfaec06"  # e.g. "sk-..." 或 None
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # e.g. "https://api.openai.com/v1" 或 None
MODEL = "qwen3-32b"
async def main():
    print("=" * 60)
    print("示例 6: 配置管理")
    print("=" * 60)
    print()
    
    # 演示不同的配置方式
    configs = []
    
    try:
        configs.append(await example_yaml_config())
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"YAML 配置示例失败: {e}\n")
    try:
        configs.append(await example_hybrid_config())
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"混合配置示例失败: {e}\n")
    
    # 使用其中一个配置进行实际对话
    if configs:
        model_config, runtime_config = configs[1]  # 使用代码配置
        
        print("【配置测试】\n")
        print("💬 用户: 你好\n")
        
        # 注意：需要有效的 API key 才能运行
        """
        async with ChatAgent(model_config, runtime_config) as agent:
            response = await agent.chat("你好")
            print(f"🤖 助手: {response}\n")
        """
        
        print("⚠️  配置已准备好，取消注释上方代码并设置 API key 即可运行\n")
    
    print("=" * 60)
    print("📝 配置最佳实践:")
    print("  1. 开发环境: 使用 YAML + 环境变量")
    print("  2. 生产环境: 使用环境变量 + 代码覆盖")
    print("  3. 测试环境: 直接代码配置")
    print("  4. 敏感信息: 始终使用环境变量")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")

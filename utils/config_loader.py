"""Configuration loader utility for examples

This module provides common functions for loading and overriding configurations
from YAML files, to be used across all example scripts.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional


def get_default_config_paths() -> Tuple[str, str]:
    """
    Get default configuration file paths
    
    Returns:
        Tuple of (model_config_path, runtime_config_path)
    """
    config_dir = Path(__file__).parent.parent / "config"
    model_config = config_dir / "default_model_config.yaml"
    runtime_config = config_dir / "default_runtime_config.yaml"
    return str(model_config), str(runtime_config)


def load_config_from_yaml(
    model_config_path: Optional[str] = None,
    runtime_config_path: Optional[str] = None,
    **overrides
) -> Tuple["ModelConfig", "RuntimeConfig"]:
    """
    Load configurations from YAML files with optional overrides
    
    Args:
        model_config_path: Path to model config YAML (default: config/default_model_config.yaml)
        runtime_config_path: Path to runtime config YAML (default: config/default_runtime_config.yaml)
        **overrides: Additional parameters to override config values
            - Model config overrides: api_key, model, temperature, max_tokens, etc.
            - Runtime config overrides: log_level, timeout, enable_debug, etc.
    
    Returns:
        Tuple of (ModelConfig, RuntimeConfig)
    
    Example:
        >>> model_cfg, runtime_cfg = load_config_from_yaml(
        ...     api_key="sk-xxx",
        ...     temperature=0.9,
        ...     log_level="DEBUG"
        ... )
    """
    # Import here to avoid circular import
    from model_config import ModelConfig
    from runtime_config import RuntimeConfig
    
    # Use default paths if not specified
    if model_config_path is None:
        model_config_path, _ = get_default_config_paths()
    if runtime_config_path is None:
        _, runtime_config_path = get_default_config_paths()
    
    # Separate overrides for model and runtime configs
    model_fields = ModelConfig.__dataclass_fields__.keys()
    runtime_fields = RuntimeConfig.__dataclass_fields__.keys()
    
    model_overrides = {k: v for k, v in overrides.items() if k in model_fields}
    runtime_overrides = {k: v for k, v in overrides.items() if k in runtime_fields}
    
    # Load configurations
    model_config = ModelConfig.from_yaml(model_config_path, **model_overrides)
    runtime_config = RuntimeConfig.from_yaml(runtime_config_path, **runtime_overrides)
    
    # Handle API key from environment variable if not set
    if model_config.api_key is None or model_config.api_key == "null":
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            model_config.api_key = env_key
    
    return model_config, runtime_config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common configuration arguments to argument parser
    
    Args:
        parser: ArgumentParser to add arguments to
    
    Returns:
        Updated ArgumentParser
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_config_args(parser)
        >>> args = parser.parse_args()
        >>> model_cfg, runtime_cfg = parse_args_to_config(args)
    """
    # Configuration files
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help="Path to runtime configuration YAML file"
    )
    
    # Common model parameter overrides
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--api-base-url", type=str, help="API base URL")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    
    # Common runtime parameter overrides
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")
    parser.add_argument("--enable-debug", action="store_true", help="Enable debug mode")
    
    return parser


def parse_args_to_config(args: argparse.Namespace) -> Tuple["ModelConfig", "RuntimeConfig"]:
    """
    Parse command line arguments to configurations
    
    Args:
        args: Parsed arguments from ArgumentParser
    
    Returns:
        Tuple of (ModelConfig, RuntimeConfig)
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_config_args(parser)
        >>> args = parser.parse_args()
        >>> model_cfg, runtime_cfg = parse_args_to_config(args)
    """
    # Build overrides dictionary
    overrides = {}
    
    # Model config overrides
    if hasattr(args, 'api_key') and args.api_key:
        overrides['api_key'] = args.api_key
    if hasattr(args, 'api_base_url') and args.api_base_url:
        overrides['api_base_url'] = args.api_base_url
    if hasattr(args, 'model') and args.model:
        overrides['model'] = args.model
    if hasattr(args, 'temperature') and args.temperature is not None:
        overrides['temperature'] = args.temperature
    if hasattr(args, 'max_tokens') and args.max_tokens is not None:
        overrides['max_tokens'] = args.max_tokens
    
    # Runtime config overrides
    if hasattr(args, 'log_level') and args.log_level:
        overrides['log_level'] = args.log_level
    if hasattr(args, 'timeout') and args.timeout is not None:
        overrides['timeout'] = args.timeout
    if hasattr(args, 'enable_debug') and args.enable_debug:
        overrides['enable_debug'] = True
        overrides['debug_save_requests'] = True
        overrides['debug_save_responses'] = True
    
    # Load configurations
    model_config_path = getattr(args, 'model_config', None)
    runtime_config_path = getattr(args, 'runtime_config', None)
    
    return load_config_from_yaml(
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        **overrides
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Simple loading
    print("Example 1: Simple loading")
    model_cfg, runtime_cfg = load_config_from_yaml()
    print(f"  Model: {model_cfg.model}, Temperature: {model_cfg.temperature}")
    print(f"  Log level: {runtime_cfg.log_level}, Timeout: {runtime_cfg.timeout}")
    
    # Example 2: With overrides
    print("\nExample 2: With overrides")
    model_cfg, runtime_cfg = load_config_from_yaml(
        temperature=0.9,
        max_tokens=2048,
        log_level="DEBUG"
    )
    print(f"  Temperature: {model_cfg.temperature}")
    print(f"  Max tokens: {model_cfg.max_tokens}")
    print(f"  Log level: {runtime_cfg.log_level}")
    
    # Example 3: Command line parsing
    print("\nExample 3: Command line parsing")
    parser = argparse.ArgumentParser()
    parser = add_config_args(parser)
    # Simulate command line args
    test_args = parser.parse_args(["--temperature", "0.8", "--log-level", "INFO"])
    model_cfg, runtime_cfg = parse_args_to_config(test_args)
    print(f"  Temperature: {model_cfg.temperature}")
    print(f"  Log level: {runtime_cfg.log_level}")

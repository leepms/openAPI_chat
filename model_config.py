"""Model configuration for OpenAI compatible API"""

from dataclasses import dataclass, asdict
from typing import Optional, Union, List
import yaml
from pathlib import Path

from exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """
    Model configuration for chat completions
    
    This config contains parameters that control the model's behavior,
    not the runtime behavior of the client itself.
    """
    
    # API connection
    api_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    # Optional request timeout (seconds)
    timeout: Optional[float] = None
    
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 1.0
    
    # Length control
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # For newer API versions
    
    # Penalties
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Other parameters
    n: int = 1  # Number of completions
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    
    # Advanced features
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    
    # Reasoning models (e.g., o1 series)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not isinstance(self.temperature, (int, float)):
            raise ConfigurationError("temperature must be a number")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError("temperature must be between 0.0 and 2.0")
        
        if not isinstance(self.top_p, (int, float)):
            raise ConfigurationError("top_p must be a number")
        if not 0.0 <= self.top_p <= 1.0:
            raise ConfigurationError("top_p must be between 0.0 and 1.0")
        
        if not isinstance(self.frequency_penalty, (int, float)):
            raise ConfigurationError("frequency_penalty must be a number")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ConfigurationError("frequency_penalty must be between -2.0 and 2.0")
        
        if not isinstance(self.presence_penalty, (int, float)):
            raise ConfigurationError("presence_penalty must be a number")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ConfigurationError("presence_penalty must be between -2.0 and 2.0")
        
        if self.n < 1:
            raise ConfigurationError("n must be at least 1")
        
        if self.reasoning_effort is not None:
            valid_efforts = ["low", "medium", "high"]
            if self.reasoning_effort not in valid_efforts:
                raise ConfigurationError(f"reasoning_effort must be one of {valid_efforts}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "ModelConfig":
        """
        Load configuration from YAML file with optional overrides
        
        Args:
            yaml_path: Path to YAML configuration file
            **overrides: Additional parameters to override YAML values
            
        Returns:
            ModelConfig instance
            
        Example:
            >>> config = ModelConfig.from_yaml("config/model.yaml", temperature=0.9)
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise ConfigurationError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Merge with overrides
        config_dict.update(overrides)
        
        # Filter only valid ModelConfig fields
        valid_fields = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return asdict(self)


# Backward compatibility alias
ChatConfig = ModelConfig

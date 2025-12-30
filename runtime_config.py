"""Runtime configuration for module behavior"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import logging
import yaml
from pathlib import Path


@dataclass
class RuntimeConfig:
    """
    Runtime configuration controlling module behavior
    
    This config controls how the module operates, not the model parameters.
    """
    
    # ==================== Logging Configuration ====================
    
    # Basic logging
    enable_logging: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # File logging
    save_logs_to_file: bool = False  # Save logs to file
    log_file_path: str = "logs/openai_chatapi.log"  # Log file path
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5  # Keep 5 backup files
    
    # ==================== HTTP Traffic Capture ====================
    
    # HTTP traffic logging (independent control)
    capture_http_traffic: bool = False  # Master switch for HTTP capture
    log_http_requests: bool = False  # Log HTTP request details (requires capture_http_traffic)
    log_http_responses: bool = False  # Log HTTP response details (requires capture_http_traffic)
    save_http_traffic_to_file: bool = False  # Save HTTP traffic to separate file
    http_traffic_file_path: str = "logs/http_traffic.log"  # HTTP traffic log file
    
    # ==================== Statistics and Monitoring ====================
    
    # Token usage tracking (independent control)
    capture_token_usage: bool = True  # Track token usage statistics
    save_token_usage_to_file: bool = False  # Save token stats to file
    token_usage_file_path: str = "logs/token_usage.log"  # Token usage log file
    
    # Latency tracking (independent control)
    capture_latency: bool = True  # Track request latency
    save_latency_to_file: bool = False  # Save latency stats to file
    latency_file_path: str = "logs/latency.log"  # Latency log file
    
    # ==================== Debug Configuration ====================
    
    enable_debug: bool = False  # Enable debug mode with extra output
    debug_save_requests: bool = False  # Save request payloads for debugging
    debug_save_responses: bool = False  # Save response payloads for debugging
    debug_output_dir: str = "debug"  # Directory for debug files
    
    # ==================== HTTP Behavior ====================
    
    timeout: int = 60  # Request timeout in seconds
    verify_ssl: bool = True  # SSL certificate verification
    max_retries: int = 0  # Number of retries on failure
    retry_delay: float = 1.0  # Delay between retries in seconds
    
    # ==================== Streaming Configuration ====================
    
    stream_chunk_callback: Optional[Callable[[str], None]] = None  # Callback for each chunk
    stream_enable_progress: bool = False  # Show progress indicators
    
    # ==================== Response Parsing ====================
    
    strict_parsing: bool = False  # Strict JSON parsing (fail on invalid)
    truncate_long_errors: bool = True  # Truncate long error messages
    max_error_length: int = 500  # Max error message length
    
    def __post_init__(self):
        """Configure logging and create directories after initialization"""
        if self.enable_logging:
            self._setup_logging()
        
        # Create directories if needed
        if self.save_logs_to_file:
            self._ensure_directory(self.log_file_path)
        if self.save_http_traffic_to_file:
            self._ensure_directory(self.http_traffic_file_path)
        if self.save_token_usage_to_file:
            self._ensure_directory(self.token_usage_file_path)
        if self.save_latency_to_file:
            self._ensure_directory(self.latency_file_path)
        if self.enable_debug and (self.debug_save_requests or self.debug_save_responses):
            Path(self.debug_output_dir).mkdir(parents=True, exist_ok=True)
    
    def _ensure_directory(self, file_path: str):
        """Ensure directory exists for file path"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        log_level = level_map.get(self.log_level.upper(), logging.INFO)
        
        # Configure logger
        logger = logging.getLogger("openai_chatapi")
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.save_logs_to_file:
            from logging.handlers import RotatingFileHandler
            
            # Ensure directory exists
            self._ensure_directory(self.log_file_path)
            
            file_handler = RotatingFileHandler(
                self.log_file_path,
                maxBytes=self.log_file_max_bytes,
                backupCount=self.log_file_backup_count
            )
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # HTTP traffic logger if enabled
        if self.capture_http_traffic and self.save_http_traffic_to_file:
            http_logger = logging.getLogger("openai_chatapi.http")
            http_logger.setLevel(logging.DEBUG)
            
            # Ensure directory exists
            self._ensure_directory(self.http_traffic_file_path)
            
            from logging.handlers import RotatingFileHandler
            http_handler = RotatingFileHandler(
                self.http_traffic_file_path,
                maxBytes=self.log_file_max_bytes,
                backupCount=self.log_file_backup_count
            )
            http_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            http_handler.setFormatter(http_formatter)
            http_logger.addHandler(http_handler)
    
    def get_retry_config(self) -> dict:
        """Get retry configuration for HTTP client"""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }
    
    def update(self, **kwargs):
        """
        Update configuration fields
        
        Args:
            **kwargs: Fields to update
            
        Example:
            >>> config = RuntimeConfig()
            >>> config.update(timeout=120, enable_debug=True)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config field: {key}")
        
        # Re-setup logging if logging-related fields changed
        logging_fields = {
            'enable_logging', 'log_level', 'save_logs_to_file', 'log_file_path',
            'capture_http_traffic', 'save_http_traffic_to_file', 'http_traffic_file_path'
        }
        if any(key in logging_fields for key in kwargs.keys()):
            if self.enable_logging:
                self._setup_logging()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != 'stream_chunk_callback'  # Exclude callable
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "RuntimeConfig":
        """
        Load configuration from YAML file with optional overrides
        
        Args:
            yaml_path: Path to YAML configuration file
            **overrides: Additional parameters to override YAML values
            
        Returns:
            RuntimeConfig instance
            
        Example:
            >>> config = RuntimeConfig.from_yaml("config/runtime.yaml", timeout=120)
        """
        from .exceptions import ConfigurationError
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise ConfigurationError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Merge with overrides
        config_dict.update(overrides)
        
        # Filter only valid RuntimeConfig fields (exclude callables)
        valid_fields = {k for k in cls.__dataclass_fields__.keys() if k != 'stream_chunk_callback'}
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
        
        config_dict = self.to_dict()
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)



@dataclass
class UsageStats:
    """Track API usage statistics"""
    
    total_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency: float = 0.0  # Total time in seconds
    errors: int = 0
    
    def add_request(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency: float = 0.0,
        error: bool = False
    ):
        """Add a request to statistics"""
        self.total_requests += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_latency += latency
        if error:
            self.errors += 1
    
    def get_average_latency(self) -> float:
        """Get average latency per request"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests
    
    def get_summary(self) -> dict:
        """Get statistics summary"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "average_latency": round(self.get_average_latency(), 3),
            "total_latency": round(self.total_latency, 3),
            "errors": self.errors,
            "success_rate": round((self.total_requests - self.errors) / self.total_requests * 100, 2) if self.total_requests > 0 else 0.0,
        }
    
    def reset(self):
        """Reset all statistics"""
        self.total_requests = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency = 0.0
        self.errors = 0
    
    def save_to_file(self, file_path: str):
        """Save statistics to file"""
        import json
        from datetime import datetime
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_summary()
        stats['timestamp'] = datetime.now().isoformat()
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(stats, ensure_ascii=False) + '\n')

"""Model manager for OpenAI compatible API"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information"""
    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "model"),
            created=data.get("created"),
            owned_by=data.get("owned_by")
        )


class ModelManager:
    """
    Model manager for fetching and managing available models
    """
    
    def __init__(
        self,
        api_base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize model manager
        
        Args:
            api_base_url: API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def list_models(self) -> List[str]:
        """
        Get list of available model IDs
        
        Returns:
            List of model ID strings
        """
        try:
            models_info = await self.list_models_detailed()
            return [model.id for model in models_info]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def list_models_detailed(self) -> List[ModelInfo]:
        """
        Get detailed information about available models
        
        Returns:
            List of ModelInfo objects
        """
        url = f"{self.api_base_url}/models"
        headers = self._get_headers()
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Parse response based on format
                models = []
                
                if "data" in data:
                    # OpenAI format: {"data": [...]}
                    for model_data in data["data"]:
                        models.append(ModelInfo.from_dict(model_data))
                elif "models" in data:
                    # Alternative format: {"models": [...]}
                    model_list = data["models"]
                    if isinstance(model_list, list):
                        for item in model_list:
                            if isinstance(item, dict):
                                models.append(ModelInfo.from_dict(item))
                            elif isinstance(item, str):
                                models.append(ModelInfo(id=item))
                elif isinstance(data, list):
                    # Direct list format
                    for item in data:
                        if isinstance(item, dict):
                            models.append(ModelInfo.from_dict(item))
                        elif isinstance(item, str):
                            models.append(ModelInfo(id=item))
                
                return models
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching models: {e.response.status_code}")
            logger.error(f"Response: {e.response.text}")
            raise
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            raise
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model
        
        Args:
            model_id: Model ID
            
        Returns:
            ModelInfo object or None if not found
        """
        url = f"{self.api_base_url}/models/{model_id}"
        headers = self._get_headers()
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                return ModelInfo.from_dict(data)
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Model not found: {model_id}")
                return None
            logger.error(f"HTTP error fetching model info: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            raise
    
    async def select_model(
        self,
        preferred_models: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select a model from available models
        
        Args:
            preferred_models: List of preferred model IDs (in order of preference)
            
        Returns:
            Selected model ID or None if no models available
        """
        try:
            available_models = await self.list_models()
            
            if not available_models:
                return None
            
            # If no preferences, return first available
            if not preferred_models:
                return available_models[0]
            
            # Try to find preferred model
            for preferred in preferred_models:
                if preferred in available_models:
                    return preferred
            
            # If no preferred model found, return first available
            return available_models[0]
            
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            return None

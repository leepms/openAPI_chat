"""
Tool Loader - Load tools from JSON definitions and dynamically import handlers

This module provides functionality to:
1. Load tool definitions from JSON files
2. Map tool definitions to their handler functions
3. Register tools with ChatAgent
4. Support multiple tool sets
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Callable, Optional, Union

from ..schema import Tool, FunctionDefinition
from . import fake_tool

logger = logging.getLogger(__name__)


class ToolLoader:
    """
    Tool loader for managing and loading tools from JSON definitions
    """
    
    def __init__(self, tool_module=None):
        """
        Initialize tool loader
        
        Args:
            tool_module: Module containing tool handler functions (default: fake_tool)
        """
        self.tool_module = tool_module or fake_tool
        self.loaded_tools: Dict[str, Tool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
    
    def load_from_json(self, json_path: Union[str, Path]) -> List[Tool]:
        """
        Load tool definitions from a JSON file
        
        Args:
            json_path: Path to JSON file containing tool definitions
        
        Returns:
            List of Tool objects
        
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Tool definition file not found: {json_path}")
        
        logger.info(f"📦 Loading tools from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            tool_defs = json.load(f)
        
        tools = []
        for tool_def in tool_defs:
            tool = self._parse_tool_definition(tool_def)
            if tool:
                tools.append(tool)
                self.loaded_tools[tool.function.name] = tool
        
        logger.info(f"✅ Loaded {len(tools)} tools: {[t.function.name for t in tools]}")
        
        return tools
    
    def load_from_multiple_json(self, json_paths: List[Union[str, Path]]) -> List[Tool]:
        """
        Load tool definitions from multiple JSON files
        
        Args:
            json_paths: List of paths to JSON files
        
        Returns:
            Combined list of Tool objects from all files
        """
        all_tools = []
        
        for json_path in json_paths:
            try:
                tools = self.load_from_json(json_path)
                all_tools.extend(tools)
            except Exception as e:
                logger.error(f"Failed to load tools from {json_path}: {e}")
        
        return all_tools
    
    def _parse_tool_definition(self, tool_def: dict) -> Optional[Tool]:
        """
        Parse a tool definition dictionary into a Tool object
        
        Args:
            tool_def: Tool definition dictionary
        
        Returns:
            Tool object or None if parsing fails
        """
        try:
            func_def = tool_def.get("function", {})
            
            function = FunctionDefinition(
                name=func_def.get("name"),
                description=func_def.get("description"),
                parameters=func_def.get("parameters")
            )
            
            tool = Tool(
                type=tool_def.get("type", "function"),
                function=function
            )
            
            return tool
        
        except Exception as e:
            logger.error(f"Failed to parse tool definition: {e}")
            return None
    
    def register_handlers(self, tools: List[Tool]) -> Dict[str, Callable]:
        """
        Register handler functions for loaded tools
        
        Args:
            tools: List of Tool objects to register handlers for
        
        Returns:
            Dictionary mapping tool names to handler functions
        """
        handlers = {}
        
        for tool in tools:
            func_name = tool.function.name
            
            # Try to get handler from tool module
            handler = getattr(self.tool_module, func_name, None)
            
            if handler and callable(handler):
                handlers[func_name] = handler
                self.tool_handlers[func_name] = handler
                logger.debug(f"✅ Registered handler for: {func_name}")
            else:
                logger.warning(f"⚠️  No handler found for: {func_name}")
        
        logger.info(f"📋 Registered {len(handlers)}/{len(tools)} tool handlers")
        
        return handlers
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a loaded tool by name
        
        Args:
            name: Tool function name
        
        Returns:
            Tool object or None if not found
        """
        return self.loaded_tools.get(name)
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """
        Get a tool handler by name
        
        Args:
            name: Tool function name
        
        Returns:
            Handler function or None if not found
        """
        return self.tool_handlers.get(name)
    
    def list_loaded_tools(self) -> List[str]:
        """
        List all loaded tool names
        
        Returns:
            List of tool names
        """
        return list(self.loaded_tools.keys())
    
    def clear(self):
        """Clear all loaded tools and handlers"""
        self.loaded_tools.clear()
        self.tool_handlers.clear()
        logger.debug("🗑️  Cleared all loaded tools")


def load_tools_for_agent(
    agent,
    json_paths: Union[str, Path, List[Union[str, Path]]],
    tool_module=None
) -> int:
    """
    Convenience function to load and register tools with a ChatAgent
    
    Args:
        agent: ChatAgent instance to register tools with
        json_paths: Single path or list of paths to tool JSON files
        tool_module: Module containing tool handlers (default: fake_tool)
    
    Returns:
        Number of tools successfully registered
    
    Example:
        >>> from openai_chatapi import ChatAgent
        >>> from openai_chatapi.tools.tool_loader import load_tools_for_agent
        >>> 
        >>> agent = ChatAgent()
        >>> count = load_tools_for_agent(agent, "tools/fake_tool.json")
        >>> print(f"Loaded {count} tools")
    """
    loader = ToolLoader(tool_module)
    
    # Handle single path or list of paths
    if isinstance(json_paths, (str, Path)):
        json_paths = [json_paths]
    
    # Load tools from JSON files
    tools = loader.load_from_multiple_json(json_paths)
    
    # Register handlers
    handlers = loader.register_handlers(tools)
    
    # Register with agent
    registered_count = 0
    for tool in tools:
        func_name = tool.function.name
        handler = handlers.get(func_name)
        
        if handler:
            agent.register_tool(tool, handler)
            registered_count += 1
    
    logger.info(f"🎉 Registered {registered_count} tools with ChatAgent")
    
    return registered_count


def get_builtin_tool_paths() -> Dict[str, Path]:
    """
    Get paths to built-in tool definition JSON files
    
    Returns:
        Dictionary mapping tool set names to their JSON file paths
    """
    tools_dir = Path(__file__).parent
    
    return {
        "all": tools_dir / "fake_tool.json",
    }


def load_builtin_tools(agent, tool_set: str = "all") -> int:
    """
    Load built-in tool sets by name
    
    Args:
        agent: ChatAgent instance
        tool_set: Name of tool set to load ("all")
    
    Returns:
        Number of tools loaded
    
    Example:
        >>> agent = ChatAgent()
        >>> load_builtin_tools(agent, "all")  # Load all tools
    """
    tool_paths = get_builtin_tool_paths()
    
    if tool_set not in tool_paths:
        logger.error(f"Unknown tool set: {tool_set}. Available: {list(tool_paths.keys())}")
        return 0
    
    return load_tools_for_agent(agent, tool_paths[tool_set])

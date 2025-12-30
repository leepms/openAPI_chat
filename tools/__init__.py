"""Tools package for OpenAI ChatAPI - Simple test tools"""

from .fake_tool import (
    web_search,
    calculate,
    get_weather,
    read_file,
    get_tool_function,
    list_available_tools,
    TOOL_FUNCTIONS,
)

from .tool_loader import (
    ToolLoader,
    load_tools_for_agent,
    load_builtin_tools,
    get_builtin_tool_paths,
)

__all__ = [
    # Tool functions
    "web_search",
    "calculate",
    "get_weather",
    "read_file",
    "get_tool_function",
    "list_available_tools",
    "TOOL_FUNCTIONS",
    
    # Tool loader
    "ToolLoader",
    "load_tools_for_agent",
    "load_builtin_tools",
    "get_builtin_tool_paths",
]

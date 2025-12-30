"""
Fake Tool Library - Simple simulated tools for testing local APIs

This module provides 4 essential test tools:
- web_search: Information retrieval simulation
- calculate: Mathematical computation
- get_weather: API call simulation  
- read_file: I/O operation simulation
"""

import time
import random
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Simulate web search
    
    Args:
        query: Search query
        max_results: Max results to return
    
    Returns:
        Search results
    """
    logger.info(f"🔍 Searching: '{query}'")
    time.sleep(random.uniform(0.5, 1.0))
    
    results = [
        {
            "title": f"Result {i+1}: {query}",
            "url": f"https://example.com/{i+1}",
            "snippet": f"Simulated content about {query}..."
        }
        for i in range(min(max_results, 3))
    ]
    
    logger.info(f"✅ Found {len(results)} results")
    return {"query": query, "results": results}


def calculate(expression: str) -> Dict[str, Any]:
    """
    Simulate mathematical calculation
    
    Args:
        expression: Math expression
    
    Returns:
        Calculation result
    """
    logger.info(f"🔢 Calculating: {expression}")
    time.sleep(random.uniform(0.1, 0.3))
    
    result = random.uniform(0, 1000)
    
    logger.info(f"✅ Result: {result:.2f}")
    return {"expression": expression, "result": round(result, 2)}


def get_weather(location: str, units: str = "metric") -> Dict[str, Any]:
    """
    Simulate weather API query
    
    Args:
        location: City name
        units: Temperature units
    
    Returns:
        Weather information
    """
    logger.info(f"🌤️  Getting weather: {location}")
    time.sleep(random.uniform(0.3, 0.8))
    
    temp = random.randint(15, 30) if units == "metric" else random.randint(60, 85)
    condition = random.choice(["Sunny", "Cloudy", "Rainy", "Clear"])
    
    logger.info(f"✅ {temp}°{'C' if units == 'metric' else 'F'}, {condition}")
    
    return {
        "location": location,
        "temperature": temp,
        "condition": condition,
        "humidity": random.randint(40, 80)
    }


def read_file(file_path: str) -> Dict[str, Any]:
    """
    Simulate reading a file
    
    Args:
        file_path: Path to file
    
    Returns:
        File content
    """
    logger.info(f"📄 Reading: {file_path}")
    time.sleep(random.uniform(0.2, 0.5))
    
    content = f"Simulated content from {file_path}\n" * random.randint(3, 10)
    
    logger.info(f"✅ Read {len(content)} bytes")
    return {"file_path": file_path, "content": content[:200]}


# Tool Registry
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculate": calculate,
    "get_weather": get_weather,
    "read_file": read_file,
}


def get_tool_function(name: str):
    """Get tool function by name"""
    return TOOL_FUNCTIONS.get(name)


def list_available_tools():
    """List all available tool names"""
    return list(TOOL_FUNCTIONS.keys())

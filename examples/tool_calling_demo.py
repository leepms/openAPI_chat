"""
Tool Calling Example - Demonstrates LLM function calling with fake tools

This example shows how to:
1. Load tool definitions from JSON files
2. Register tools with ChatAgent
3. Let the LLM decide when to call tools
4. Execute tools and feed results back to LLM
5. Get final response after tool execution
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai_chatapi import ChatAgent, ModelConfig, RuntimeConfig, load_tools_for_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def tool_calling_demo():
    """Demonstrate tool calling with fake tools"""
    
    print("=" * 70)
    print("Tool Calling Demo - Fake Tools")
    print("=" * 70)
    print()
    
    # Create agent with configuration
    model_config = ModelConfig(
        api_base_url="https://api.openai.com/v1",
        api_key="your-api-key-here",  # Replace with actual key or set OPENAI_API_KEY env var
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000,
    )
    
    runtime_config = RuntimeConfig(
        enable_logging=True,
        log_level="INFO",
        enable_debug=True,
        capture_token_usage=True,
    )
    
    async with ChatAgent(model_config, runtime_config) as agent:
        
        # ========== Example 1: Load all tools ==========
        print("\n📦 Example 1: Loading all available tools\n")
        
        tools_dir = Path(__file__).parent.parent / "tools"
        tool_file = tools_dir / "fake_tool.json"
        
        tool_count = load_tools_for_agent(agent, tool_file)
        print(f"✅ Loaded {tool_count} tools\n")
        
        # Ask a question that requires tool usage
        print("💬 User: What's the weather like in Beijing?\n")
        
        response = await agent.chat(
            "What's the weather like in Beijing?",
            auto_execute_tools=True,  # Automatically execute tool calls
            max_tool_iterations=5,
        )
        
        print(f"🤖 Assistant: {response}\n")
        print("-" * 70)
        
        # ========== Example 2: Multiple tool calls ==========
        print("\n📦 Example 2: Multiple tool calls in sequence\n")
        
        agent.clear_history(keep_system=False)
        
        print("💬 User: Search for 'Python programming' and calculate 5 * 8\n")
        
        response = await agent.chat(
            "Search the web for 'Python programming' and then calculate 5 * 8",
            auto_execute_tools=True,
            max_tool_iterations=5,
        )
        
        print(f"🤖 Assistant: {response}\n")
        print("-" * 70)
        
        # ========== Example 3: Load specific tool sets ==========
        print("\n📦 Example 3: Loading specific tool sets\n")
        
        agent.clear_tools()
        agent.clear_history(keep_system=False)
        
        # Load only file tools
        file_tools = tools_dir / "file_tools.json"
        api_tools = tools_dir / "api_tools.json"
        
        tool_count = load_tools_for_agent(agent, [file_tools, api_tools])
        print(f"✅ Loaded {tool_count} tools from multiple files\n")
        
        print("💬 User: Read the file at /tmp/data.txt and translate it to Chinese\n")
        
        response = await agent.chat(
            "Read the file at /tmp/data.txt and translate the content to Chinese",
            auto_execute_tools=True,
            max_tool_iterations=5,
        )
        
        print(f"🤖 Assistant: {response}\n")
        print("-" * 70)
        
        # ========== Example 4: Complex multi-step task ==========
        print("\n📦 Example 4: Complex multi-step workflow\n")
        
        agent.clear_history(keep_system=False)
        
        print("💬 User: Get Beijing weather, stock price for AAPL, and send notification\n")
        
        response = await agent.chat(
            "Please do the following: 1) Get the weather in Beijing, "
            "2) Get the stock price for AAPL, "
            "3) Send me a notification with both pieces of information",
            auto_execute_tools=True,
            max_tool_iterations=10,
        )
        
        print(f"🤖 Assistant: {response}\n")
        print("-" * 70)
        
        # ========== Example 5: Manual tool execution (no auto) ==========
        print("\n📦 Example 5: Manual tool execution control\n")
        
        agent.clear_history(keep_system=False)
        
        print("💬 User: Calculate 123 * 456\n")
        
        # This will return immediately if LLM wants to call a tool
        response = await agent.chat(
            "Calculate 123 * 456",
            auto_execute_tools=False,  # Don't automatically execute
            max_tool_iterations=1,
        )
        
        print(f"🤖 Assistant (may be empty if tool call needed): {response}\n")
        
        # Check if there are pending tool calls
        if agent.messages[-1].tool_calls:
            print("⚠️  LLM wants to call tools, but auto_execute_tools=False")
            print("Tool calls:", [tc.function.name for tc in agent.messages[-1].tool_calls])
        
        print("-" * 70)
        
        # ========== Statistics ==========
        print("\n📊 Usage Statistics\n")
        stats = agent.get_stats()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average latency: {stats['average_latency']:.2f}s")
        print(f"Error rate: {stats['error_count']}/{stats['total_requests']}")
        
        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70)

"""
ARCHIVED: tool_calling_demo.py

Full tool-calling demo has been archived. Use `tests/test_all_features.py`
for fake tool testing or the `examples/example_5_tool_calling.py` stub.
"""

print("tool_calling_demo.py archived. Use tests/test_all_features.py for live demos.")
async def simple_tool_demo():
    """Simple demonstration of tool calling"""
    
    print("\n" + "=" * 70)
    print("Simple Tool Calling Demo")
    print("=" * 70 + "\n")
    
    # Create agent
    model_config = ModelConfig(
        api_key="your-api-key-here",
        model="gpt-4o-mini",
    )
    
    async with ChatAgent(model_config) as agent:
        # Load tools
        tools_dir = Path(__file__).parent.parent / "tools"
        load_tools_for_agent(agent, tools_dir / "fake_tool.json")
        
        # Simple query
        print("💬 User: What's the weather in Tokyo?\n")
        response = await agent.chat(
            "What's the weather in Tokyo?",
            auto_execute_tools=True
        )
        print(f"🤖 Assistant: {response}\n")


if __name__ == "__main__":
    # Check if API key is set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it or modify the code to include your API key\n")
        print("Running demo with placeholder key (will use fake tools only)...\n")
    
    # Run full demo
    try:
        asyncio.run(tool_calling_demo())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

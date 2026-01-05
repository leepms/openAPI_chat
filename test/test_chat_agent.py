"""
Test script for OpenAI compatible chat agent
Tests all functionality without requiring actual API access
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
print("=" * 60)
print("Testing OpenAI Chat API Client")
print("=" * 60)

def test_imports():
    """Test 1: Import modules"""
    print("\n[Test 1] Testing imports...")
    try:
        from chat_agent import ChatAgent
        from model_config import ChatConfig
        from schema import (
            ChatMessage,
            ChatCompletionRequest,
            MessageContentText,
            MessageContentImageUrl,
            MessageImageUrl,
        )
        from utils.media_utils import (
            encode_image_to_base64,
            create_text_content,
            create_image_content,
            create_user_message,
            create_system_message,
            create_assistant_message,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test 2: Configuration"""
    print("\n[Test 2] Testing configuration...")
    try:
        from model_config import ChatConfig
        from exceptions import ConfigurationError
        
        # Test default config
        config = ChatConfig()
        assert config.api_base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        print("✓ Default config works")
        
        # Test custom config
        config = ChatConfig(
            api_base_url="http://localhost:8000",
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=1.0,
            max_tokens=100,
            timeout=30,
            stream=True
        )
        assert config.api_base_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 1.0
        print("✓ Custom config works")
        
        # Test validation
        try:
            config = ChatConfig(temperature=3.0)  # Should fail
            print("✗ Temperature validation failed")
            return False
        except ConfigurationError:
            print("✓ Temperature validation works")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema():
    """Test 3: Schema objects"""
    print("\n[Test 3] Testing schema objects...")
    try:
        from schema import (
            ChatMessage,
            ChatCompletionRequest,
            MessageContentText,
            MessageImageUrl,
        )
        
        # Test ChatMessage with string content
        msg1 = ChatMessage(role="user", content="Hello")
        assert msg1.role == "user"
        assert msg1.content == "Hello"
        msg_dict = msg1.to_dict()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello"
        print("✓ ChatMessage with text works")
        
        # Test ChatMessage with list content
        text_content = MessageContentText(type="text", text="Hello")
        msg2 = ChatMessage(role="user", content=[text_content])
        msg_dict = msg2.to_dict()
        assert msg_dict["role"] == "user"
        assert isinstance(msg_dict["content"], list)
        print("✓ ChatMessage with content list works")
        
        # Test ChatCompletionRequest
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[msg1],
            temperature=0.5
        )
        req_dict = request.to_dict()
        assert req_dict["model"] == "gpt-4"
        assert req_dict["temperature"] == 0.5
        assert len(req_dict["messages"]) == 1
        print("✓ ChatCompletionRequest works")
        
        return True
    except Exception as e:
        print(f"✗ Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test 4: Utility functions"""
    print("\n[Test 4] Testing utility functions...")
    try:
        from utils.media_utils import (
            create_text_content,
            create_user_message,
            create_system_message,
            create_assistant_message,
        )
        
        # Test create_text_content
        text_content = create_text_content("Hello world")
        assert text_content.type == "text"
        assert text_content.text == "Hello world"
        print("✓ create_text_content works")
        
        # Test create_user_message (text only)
        user_msg = create_user_message("What is AI?")
        assert user_msg.role == "user"
        assert user_msg.content == "What is AI?"
        print("✓ create_user_message (text) works")
        
        # Test create_system_message
        sys_msg = create_system_message("You are helpful")
        assert sys_msg.role == "system"
        assert sys_msg.content == "You are helpful"
        print("✓ create_system_message works")
        
        # Test create_assistant_message
        asst_msg = create_assistant_message("I can help")
        assert asst_msg.role == "assistant"
        assert asst_msg.content == "I can help"
        print("✓ create_assistant_message works")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_encoding():
    """Test 5: Image encoding"""
    print("\n[Test 5] Testing image encoding...")
    try:
        from utils.media_utils import encode_image_to_base64, create_image_content
        import tempfile
        from PIL import Image
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            test_image_path = f.name
        
        try:
            # Test encoding
            base64_uri = encode_image_to_base64(test_image_path)
            assert base64_uri.startswith("data:image/")
            assert "base64," in base64_uri
            print("✓ Image encoding works")
            
            # Test create_image_content
            img_content = create_image_content(test_image_path)
            assert img_content.type == "image_url"
            assert img_content.image_url.url.startswith("data:image/")
            print("✓ create_image_content works")
            
        finally:
            # Cleanup
            Path(test_image_path).unlink(missing_ok=True)
        
        return True
    except ImportError:
        print("⚠ PIL not installed, skipping image tests (optional)")
        return True
    except Exception as e:
        print(f"✗ Image encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_initialization():
    """Test 6: Agent initialization"""
    print("\n[Test 6] Testing agent initialization...")
    try:
        from chat_agent import ChatAgent
        from model_config import ChatConfig
        
        # Test with default config
        agent1 = ChatAgent()
        assert agent1.config is not None
        assert len(agent1.messages) == 0
        await agent1.close()
        print("✓ Agent with default config works")
        
        # Test with custom config
        config = ChatConfig(
            api_base_url="http://localhost:8000",
            model="test-model"
        )
        agent2 = ChatAgent(config)
        assert agent2.config.api_base_url == "http://localhost:8000"
        assert agent2.config.model == "test-model"
        await agent2.close()
        print("✓ Agent with custom config works")
        
        # Test context manager
        async with ChatAgent(config) as agent3:
            assert agent3.config is not None
        print("✓ Agent context manager works")
        
        return True
    except Exception as e:
        print(f"✗ Agent initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_methods():
    """Test 7: Agent methods"""
    print("\n[Test 7] Testing agent methods...")
    try:
        from chat_agent import ChatAgent
        from model_config import ChatConfig
        
        config = ChatConfig()
        async with ChatAgent(config) as agent:
            # Test set_system_prompt
            agent.set_system_prompt("You are a helpful assistant")
            assert len(agent.messages) == 1
            assert agent.messages[0].role == "system"
            print("✓ set_system_prompt works")
            
            # Test add_message
            from utils.media_utils import create_user_message
            user_msg = create_user_message("Hello")
            agent.add_message(user_msg)
            assert len(agent.messages) == 2
            print("✓ add_message works")
            
            # Test clear_history (keep system)
            agent.clear_history(keep_system=True)
            assert len(agent.messages) == 1
            assert agent.messages[0].role == "system"
            print("✓ clear_history (keep_system=True) works")
            
            # Test clear_history (clear all)
            agent.clear_history(keep_system=False)
            assert len(agent.messages) == 0
            print("✓ clear_history (keep_system=False) works")
        
        return True
    except Exception as e:
        print(f"✗ Agent methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_request_building():
    """Test 8: Request building"""
    print("\n[Test 8] Testing request building...")
    try:
        from chat_agent import ChatAgent
        from model_config import ChatConfig
        from utils.media_utils import create_user_message
        
        config = ChatConfig(model="gpt-4", temperature=0.8)
        async with ChatAgent(config) as agent:
            agent.set_system_prompt("Test prompt")
            agent.add_message(create_user_message("Test message"))
            
            # Build request
            request = agent._build_request()
            assert request.model == "gpt-4"
            assert request.temperature == 0.8
            assert len(request.messages) == 2
            print("✓ Request building works")
            
            # Test request serialization
            req_dict = request.to_dict()
            assert "model" in req_dict
            assert "messages" in req_dict
            assert req_dict["model"] == "gpt-4"
            assert len(req_dict["messages"]) == 2
            print("✓ Request serialization works")
            
            # Test with custom parameters
            request2 = agent._build_request(temperature=1.0, max_tokens=100)
            assert request2.temperature == 1.0
            assert request2.max_tokens == 100
            print("✓ Request with custom params works")
        
        return True
    except Exception as e:
        print(f"✗ Request building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_stream_tool_execution():
    """Test streaming tool execution with chunked function arguments"""
    print("\n[Test Stream Tool] Testing streamed tool execution...")
    try:
        from chat_agent import ChatAgent
        from model_config import ChatConfig
        from schema import (
            ChatCompletionChunkChoice,
            ChatCompletionChunkDelta,
        )
        from schema import Tool, FunctionDefinition

        config = ChatConfig()

        async def fake_send(request):
            # Simulate two chunks: first contains partial arguments, second completes and sets finish_reason
            # First chunk: delta with tool call partial
            tc1 = {
                "id": "t1",
                "function": {"name": "echo_tool", "arguments": '{"x": 1,'}
            }

            choice1 = ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(content="Hello", tool_calls=[tc1]),
                finish_reason=None
            )

            args2 = '"y": 2}'
            tc2 = {
                "id": "t1",
                "function": {"name": "echo_tool", "arguments": args2}
            }

            choice2 = ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(content=" World", tool_calls=[tc2]),
                finish_reason="function_call"
            )

            # Yield choices as the streaming function would
            yield choice1
            yield choice2

        async with ChatAgent(config) as agent:
            # Register a simple tool handler
            def echo_tool(x=None, y=None):
                return {"x": x, "y": y}

            tool = Tool(function=FunctionDefinition(name="echo_tool", parameters={"type": "object", "properties": {"x": {}, "y": {}}}))
            agent.register_tool(tool, echo_tool)

            # Patch the agent's _send_stream_request to our fake
            agent._send_stream_request = fake_send

            collected = []
            async for chunk in agent.chat_stream("test", auto_execute_tools=True):
                collected.append(chunk)

            # After stream, a tool message should be in history
            tool_msgs = [m for m in agent.messages if m.role == 'tool']
            assert len(tool_msgs) >= 1
            print("✓ Streamed tool executed and result injected into history")
            return True
    except Exception as e:
        print(f"✗ Stream tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_parsing():
    """Test 9: Response parsing"""
    print("\n[Test 9] Testing response parsing...")
    try:
        from schema import ChatCompletionResponse, ChatCompletionChunk
        
        # Test ChatCompletionResponse parsing
        response_data = {
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        response = ChatCompletionResponse.from_dict(response_data)
        assert response.id == "test-123"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.usage is not None
        assert response.usage.total_tokens == 15
        print("✓ ChatCompletionResponse parsing works")
        
        # Test ChatCompletionChunk parsing
        chunk_data = {
            "id": "test-456",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hi"
                },
                "finish_reason": None
            }]
        }
        
        chunk = ChatCompletionChunk.from_dict(chunk_data)
        assert chunk.id == "test-456"
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta.content == "Hi"
        print("✓ ChatCompletionChunk parsing works")
        
        return True
    except Exception as e:
        print(f"✗ Response parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    results = []
    
    # Synchronous tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Schema Objects", test_schema()))
    results.append(("Utility Functions", test_utils()))
    results.append(("Image Encoding", test_image_encoding()))
    
    # Asynchronous tests
    results.append(("Agent Initialization", await test_agent_initialization()))
    results.append(("Agent Methods", await test_agent_methods()))
    results.append(("Request Building", await test_request_building()))
    results.append(("Stream Tool Execution", await test_stream_tool_execution()))
    results.append(("Response Parsing", test_response_parsing()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

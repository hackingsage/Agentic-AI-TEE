import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import enclave modules
project_root = Path(r"c:\Users\AYUSH\Downloads\Agentic-AI-TEE")
sys.path.insert(0, str(project_root))

from enclave.agent.llm_client import GroqClient
from enclave.agent.models import Message
from enclave.tools.base import ToolRegistry
from enclave.tools.list_dir import ListDirTool
from enclave.tools.grep_search import GrepSearchTool
from enclave.tools.bash_tool import BashTool

async def test_groq():
    api_key = "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"
    client = GroqClient(model="llama-3.3-70b-versatile", api_key=api_key)
    
    # We will register our tools to build tool definitions
    registry = ToolRegistry()
    registry.register(ListDirTool(project_root=project_root))
    registry.register(GrepSearchTool(project_root=project_root))
    registry.register(BashTool(cwd=project_root))
    
    tools = registry.build_tool_definitions()
    
    system_prompt = "You are a coding assistant."
    messages = [Message(role="user", content="tell me about my system")]
    
    print("Calling Groq API...")
    try:
        response = await client.call(
            system=system_prompt,
            messages=messages,
            tools=tools,
        )
        print("Success!")
        print("Response:", response.text)
    except Exception as e:
        print("Failed with exception:", e)
        # If it has response attribute, print its text
        if hasattr(e, "response"):
            print("Response status:", e.response.status_code)
            print("Response body:", e.response.text)

if __name__ == "__main__":
    asyncio.run(test_groq())

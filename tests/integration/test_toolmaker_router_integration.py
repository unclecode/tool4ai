# tests/integration/test_toolmaker_router_integration.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock
import uuid
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import SubQueryResponse, SubQuery
from tool4ai.toolmakers.openai_maker import OpenAIToolMaker

@pytest.fixture(scope="module")
def sample_toolkit():
    toolkit = Toolkit()
    tools = [
        Tool(name="tool1", schema={"type": "object", "properties": {"arg1": {"type": "string"}}}, description="This is description for tool 1"),
        Tool(name="tool2", schema={"type": "object", "properties": {"arg2": {"type": "integer"}}}, description="This is description for tool 2"),
    ]
    for tool in tools:
        toolkit.add_tool(tool)
    return toolkit

def test_toolmaker_router_integration(sample_toolkit):
    tool_maker = OpenAIToolMaker()
    router = Router(sample_toolkit, tool_maker)
    
    # Test routing
    graph = router.route("Use tool1 to do something")
      
    # Verify that the graph was created correctly
    assert len(graph.sub_queries) == 1
    assert graph.sub_queries[0].tool == "tool1"
    
    

if __name__ == "__main__":
    pytest.main([__file__])
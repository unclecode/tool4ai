# tests/test_router.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# tests/test_router.py

import pytest
from unittest.mock import MagicMock, patch
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import SubQueryResponse, SubQuery
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph

@pytest.fixture
def sample_toolkit():
    toolkit = Toolkit()
    tool = Tool(
        name="sample_tool",
        schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        description="A sample tool"
    )
    toolkit.add_tool(tool)
    return toolkit

@pytest.fixture
def mock_tool_maker():
    return MagicMock()

def test_router_initialization(sample_toolkit, mock_tool_maker):
    router = Router(sample_toolkit, mock_tool_maker)
    assert isinstance(router.toolkit, Toolkit)
    assert router.tool_maker == mock_tool_maker

def test_router_gen_subquery(sample_toolkit, mock_tool_maker):
    router = Router(sample_toolkit, mock_tool_maker)
    mock_response = SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Test query", task="Test task", tool="sample_tool")
    ])
    mock_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    
    mock_tool_maker.completion.return_value = (mock_response, mock_usage)

    result, usage = router.gen_subquery("Test query")
    
    assert isinstance(result, SubQueryResponse)
    assert len(result.sub_queries) == 1
    # assert usage == mock_usage

def test_router_route(sample_toolkit, mock_tool_maker):
    router = Router(sample_toolkit, mock_tool_maker)
    mock_response = SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Test query", task="Test task", tool="sample_tool")
    ])
    mock_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    
    mock_tool_maker.completion.return_value = (mock_response, mock_usage)

    result = router.route("Test query")
    
    assert isinstance(result, ToolDependencyGraph)
    assert len(result.sub_queries) == 1

def test_router_token_usage(sample_toolkit, mock_tool_maker):
    router = Router(sample_toolkit, mock_tool_maker)
    mock_response = SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Test query", task="Test task", tool="sample_tool")
    ])
    mock_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    
    mock_tool_maker.completion.return_value = (mock_response, mock_usage)

    router.route("Test query")
    
    total_usage = router.get_total_token_usage()
    # assert total_usage["prompt_tokens"] == 10
    # assert total_usage["completion_tokens"] == 20
    # assert total_usage["total_tokens"] == 30
# tests/integration/test_router_graph_integration.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import SubQueryResponse, SubQuery
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph

@pytest.fixture(scope="module")
def sample_toolkit():
    toolkit = Toolkit()
    tools = [
        Tool(name="tool1", schema={"type": "object", "properties": {"arg1": {"type": "string"}}}, description="The first tool to do cool things"),
        Tool(name="tool2", schema={"type": "object", "properties": {"arg2": {"type": "integer"}}}, description="The second tool to do even cooler things"),
    ]
    for tool in tools:
        toolkit.add_tool(tool)
    return toolkit

@pytest.fixture(scope="module")
def mock_tool_maker():
    tool_maker = MagicMock()
    tool_maker.completion.return_value = (
        SubQueryResponse(sub_queries=[
            SubQuery(index=0, sub_query="Query 1", task="Task 1", tool="tool1"),
            SubQuery(index=1, sub_query="Query 2", task="Task 2", tool="tool2", dependent_on=0),
        ]),
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )
    return tool_maker

def test_router_graph_integration(sample_toolkit, mock_tool_maker):
    router = Router(sample_toolkit, mock_tool_maker)
    
    # Reset mock counts
    mock_tool_maker.completion.reset_mock()
    
    # Test routing
    graph = router.route("Please use tool1 and after then when you are done use tool2")
    
    # Verify graph structure
    assert isinstance(graph, ToolDependencyGraph)
    assert len(graph.sub_queries) == 2
    assert 0 in graph.sub_queries
    assert 1 in graph.sub_queries
    assert graph.sub_queries[0].tool == "tool1"
    assert graph.sub_queries[1].tool == "tool2"
    
    # Verify dependencies
    assert 1 in graph.dependency_map
    assert 0 in graph.dependency_map[1]
    
    # Verify execution order
    execution_order = graph.get_execution_order()
    assert len(execution_order) == 2
    assert execution_order[0] == [0]
    assert execution_order[1] == [1]
       
if __name__ == "__main__":
    pytest.main([__file__])
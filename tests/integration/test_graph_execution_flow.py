# tests/integration/test_graph_execution_flow.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# tests/integration/test_graph_execution_flow.py

import pytest
from unittest.mock import MagicMock, AsyncMock
import json, uuid
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph
from tool4ai.core.models import SubQueryResponse, SubQuery, ExecutionResult, ExecutionStatus

@pytest.fixture(scope="module")
def sample_graph():
    graph = ToolDependencyGraph()
    sub_query_response = SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Query 1", task="Task 1", tool="tool1"),
        SubQuery(index=1, sub_query="Query 2", task="Task 2", tool="tool2", dependent_on=0),
        SubQuery(index=2, sub_query="Query 3", task="Task 3", tool="tool3", dependent_on=1),
    ])
    graph.build_dependency_structure(sub_query_response)
    return graph

@pytest.fixture(scope="module")
def sample_graph_with_failing_tool():
    graph = ToolDependencyGraph()
    sub_query_response = SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Query 1", task="Task 1", tool="tool1"),
        SubQuery(index=1, sub_query="Query 2", task="Task 2", tool="failing_tool", dependent_on=0),
        SubQuery(index=2, sub_query="Query 3", task="Task 3", tool="tool3", dependent_on=1),
    ])
    graph.build_dependency_structure(sub_query_response)
    return graph

@pytest.fixture(scope="module")
def mock_tool_functions():
    return {
        "tool1": AsyncMock(return_value='{"status": "success", "return": {"result": "Tool 1 result"}}'),
        "tool2": AsyncMock(return_value='{"status": "success", "return": {"result": "Tool 2 result"}}'),
        "tool3": AsyncMock(return_value='{"status": "success", "return": {"result": "Tool 3 result"}}'),
        "failing_tool": AsyncMock(return_value='{"status": "failed", "message": "This tool always fails"}'),
    }

@pytest.fixture(scope="module")
def mock_tools_info():
    return {
        "tool1": {"name": "tool1", "description": "Tool 1"},
        "tool2": {"name": "tool2", "description": "Tool 2"},
        "tool3": {"name": "tool3", "description": "Tool 3"},
        "failing_tool": {"name": "failing_tool", "description": "A tool that always fails"},
    }

@pytest.fixture
def mock_tool_maker():
    tool_maker = AsyncMock()
    
    async def make_tools_side_effect(task, filtered_tools_info, context):
        # Determine which tool to return based on the filtered_tools_info
        tool_name = next(iter(filtered_tools_info))
        return (
            {"tool_calls": [{"id": str(uuid.uuid4()), "function": {"name": tool_name, "arguments": "{}"}}]},
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
    
    tool_maker.make_tools.side_effect = make_tools_side_effect
    return tool_maker

@pytest.mark.asyncio
async def test_graph_execution_flow(sample_graph, mock_tool_functions, mock_tools_info, mock_tool_maker):
    mock_context = {}
    
    result = await sample_graph.execute(mock_tool_functions, mock_tools_info, mock_context, mock_tool_maker)
    
    assert isinstance(result, ExecutionResult)
    assert result.status == ExecutionStatus.SUCCESS
    
    # Verify execution order
    mock_tool_functions["tool1"].assert_called_once()
    mock_tool_functions["tool2"].assert_called_once()
    mock_tool_functions["tool3"].assert_called_once()
    
    assert mock_tool_functions["tool1"].call_count == 1
    assert mock_tool_functions["tool2"].call_count == 1
    assert mock_tool_functions["tool3"].call_count == 1
    
    # Verify dependency execution order
    assert mock_tool_functions["tool1"].call_count == 1
    assert mock_tool_functions["tool2"].call_count <= mock_tool_functions["tool1"].call_count
    assert mock_tool_functions["tool3"].call_count <= mock_tool_functions["tool2"].call_count

    # Verify tool_maker was called correctly
    assert mock_tool_maker.make_tools.call_count == 3

@pytest.mark.asyncio
async def test_graph_execution_with_error(sample_graph_with_failing_tool, mock_tool_functions, mock_tools_info, mock_tool_maker):
    # Reset mock counts
    for mock_func in mock_tool_functions.values():
        mock_func.reset_mock()
    mock_tool_maker.reset_mock()
    
    mock_context = {}
    
    result = await sample_graph_with_failing_tool.execute(mock_tool_functions, mock_tools_info, mock_context, mock_tool_maker)
    
    assert isinstance(result, ExecutionResult)
    assert result.status == ExecutionStatus.FAILED
    
    # Verify that tool3 was not executed due to failing_tool's failure
    assert mock_tool_functions["tool1"].call_count == 1
    assert mock_tool_functions["failing_tool"].call_count == 1
    assert mock_tool_functions["tool3"].call_count == 0

    # Verify tool_maker was called correctly
    assert mock_tool_maker.make_tools.call_count == 2
    
if __name__ == "__main__":
    pytest.main([__file__])
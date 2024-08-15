# tests/test_tool_dependency_graph.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# tests/test_tool_dependency_graph.py

import pytest
from unittest.mock import MagicMock, patch
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph
from tool4ai.core.models import SubQueryResponse, SubQuery, ExecutionResult, ExecutionStatus

@pytest.fixture
def sample_sub_query_response():
    return SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Query 1", task="Task 1", tool="tool1"),
        SubQuery(index=1, sub_query="Query 2", task="Task 2", tool="tool2", dependent_on=0),
    ])

def test_build_dependency_structure(sample_sub_query_response):
    graph = ToolDependencyGraph()
    graph.build_dependency_structure(sample_sub_query_response)
    
    assert len(graph.sub_queries) == 2
    assert 1 in graph.dependency_map
    assert 0 in graph.dependency_map[1]
    assert 0 in graph.reverse_dependency_map

def test_get_execution_order(sample_sub_query_response):
    graph = ToolDependencyGraph()
    graph.build_dependency_structure(sample_sub_query_response)
    
    execution_order = graph.get_execution_order()
    assert len(execution_order) == 2
    assert execution_order[0] == [0]
    assert execution_order[1] == [1]

@pytest.mark.asyncio
async def test_execute(sample_sub_query_response):
    graph = ToolDependencyGraph()
    graph.build_dependency_structure(sample_sub_query_response)
    
    mock_tool_functions = {
        "tool1": MagicMock(return_value='{"status": "success", "return": {}}'),
        "tool2": MagicMock(return_value='{"status": "success", "return": {}}')
    }
    mock_tools_info = {
        "tool1": {"name": "tool1", "description": "Tool 1"},
        "tool2": {"name": "tool2", "description": "Tool 2"}
    }
    mock_context = {}
    mock_tool_maker = MagicMock()
    
    result = await graph.execute(mock_tool_functions, mock_tools_info, mock_context, mock_tool_maker)
    
    assert isinstance(result, ExecutionResult)
    # assert result.status == ExecutionStatus.SUCCESS

@pytest.mark.asyncio
async def test_save_and_load(sample_sub_query_response):
    graph = ToolDependencyGraph()
    graph.build_dependency_structure(sample_sub_query_response)
    
    await graph.save()
    
    loaded_graph = await ToolDependencyGraph.load(graph.run_id)
    
    assert loaded_graph.run_id == graph.run_id
    assert len(loaded_graph.sub_queries) == len(graph.sub_queries)
    assert loaded_graph.dependency_map == graph.dependency_map

def test_update_token_usage():
    graph = ToolDependencyGraph()
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    graph.update_token_usage(usage)
    
    total_usage = graph.get_token_usage()
    assert total_usage["prompt_tokens"] == 10
    assert total_usage["completion_tokens"] == 20
    assert total_usage["total_tokens"] == 30
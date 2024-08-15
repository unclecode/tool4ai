# tests/integration/test_storage_integration.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph
from tool4ai.core.models import SubQueryResponse, SubQuery
from tool4ai.storages import JSONStorage, LMDBStorage

@pytest.fixture(scope="module")
def sample_sub_query_response():
    return SubQueryResponse(sub_queries=[
        SubQuery(index=0, sub_query="Query 1", task="Task 1", tool="tool1"),
        SubQuery(index=1, sub_query="Query 2", task="Task 2", tool="tool2", dependent_on=0),
    ])

@pytest.mark.parametrize("storage_class", [JSONStorage, LMDBStorage])
@pytest.mark.asyncio
async def test_storage_integration(sample_sub_query_response, storage_class):
    # Initialize the graph with the specific storage
    graph = ToolDependencyGraph(storage=storage_class())
    
    # Build the graph structure
    graph.build_dependency_structure(sample_sub_query_response)
    
    # Save the graph
    await graph.save()
    
    # Load the graph
    loaded_graph = await ToolDependencyGraph.load(graph.run_id, storage=storage_class())
    
    # Verify the loaded graph
    assert loaded_graph.run_id == graph.run_id
    assert len(loaded_graph.sub_queries) == len(graph.sub_queries)
    assert loaded_graph.dependency_map == graph.dependency_map
    assert loaded_graph.reverse_dependency_map == graph.reverse_dependency_map
    
    # Verify sub_queries
    for index, sub_query in graph.sub_queries.items():
        loaded_sub_query = loaded_graph.sub_queries[index]
        assert loaded_sub_query.sub_query == sub_query.sub_query
        assert loaded_sub_query.task == sub_query.task
        assert loaded_sub_query.tool == sub_query.tool
        assert loaded_sub_query.dependent_on == sub_query.dependent_on
    
    # Clean up
    await graph.delete()

if __name__ == "__main__":
    pytest.main([__file__])
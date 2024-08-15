# tests/performance/test_tool4ai_performance.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
import asyncio
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.toolmakers.openai_maker import OpenAIToolMaker
from tool4ai.utils.config_manager import config_manager

# Helper function to create a large number of dummy tools
def create_dummy_tools(num_tools: int) -> Dict[str, Dict[str, Any]]:
    tool_schemas = {}
    for i in range(num_tools):
        tool_schemas[f"dummy_tool_{i}"] = {
            "name": f"dummy_tool_{i}",
            "description": f"Dummy tool number {i}",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": f"Argument for dummy tool {i}"}
                },
                "required": ["arg1"]
            }
        }
    return tool_schemas

# Dummy tool function
async def dummy_tool_function(arguments: Dict[str, Any]):
    await asyncio.sleep(0.1)  # Simulate some processing time
    return '{"status": "success", "return": {"result": "Dummy result"}}'

# 4.1 Load Testing
@pytest.mark.asyncio
async def test_large_number_of_tools():
    num_tools = 20
    tool_schemas = create_dummy_tools(num_tools)
    
    toolkit = Toolkit()
    for name, schema in tool_schemas.items():
        tool = Tool(name=name, schema=schema["parameters"], description=schema["description"])
        toolkit.add_tool(tool)
    
    tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
    router = Router(toolkit, tool_maker)
    
    start_time = time.time()
    graph = router.route("Use a random tool")
    end_time = time.time()
    
    assert len(toolkit.tools) == num_tools
    print(f"Time taken to route with {num_tools} tools: {end_time - start_time} seconds")

def generate_complex_query(tool_names: List[str], chain_length: int) -> str:
    if chain_length > len(tool_names):
        raise ValueError("Chain length cannot be greater than the number of tools")
    
    query_parts = []
    for i in range(chain_length):
        if i == 0:
            query_parts.append(f"First use {tool_names[i]}")
        elif i == chain_length - 1:
            query_parts.append(f"Finally, use {tool_names[i]} with the combined results of all previous tools")
        elif i == 1:
            query_parts.append(f"Then based on its result use {tool_names[i]}")
        elif i % 2 == 0:
            query_parts.append(f"After that, use {tool_names[i]} and {tool_names[i+1]} in parallel")
            i += 1  # Skip next tool as it's used in parallel
        else:
            query_parts.append(f"Then use {tool_names[i]}")
    
    return ". ".join(query_parts) + "."

@pytest.mark.asyncio
@pytest.mark.parametrize("total_tools,chain_length", [
    # (10, 5),
    (20, 10),
    # (50, 25),
    # (100, 50)
])
async def test_complex_query(total_tools: int, chain_length: int):
    toolkit = Toolkit()
    tool_schemas = create_dummy_tools(total_tools)
    for name, schema in tool_schemas.items():
        tool = Tool(name=name, schema=schema["parameters"], description=schema["description"])
        toolkit.add_tool(tool)
    
    tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
    router = Router(toolkit, tool_maker)
    
    tool_names = list(tool_schemas.keys())[:chain_length]
    complex_query = generate_complex_query(tool_names, chain_length)
    
    print(f"\nTesting complex query with {total_tools} total tools and {chain_length} chained tools")
    print(f"Generated query: {complex_query}")
    
    start_time = time.time()
    graph = router.route(complex_query)
    end_time = time.time()
    
    routing_time = end_time - start_time
    print(f"Time taken to route complex query: {routing_time:.4f} seconds")
    print(f"Number of sub-queries: {len(graph.sub_queries)}")
    
    # Optional: Execute the graph to test full pipeline
    tool_functions = {tool.name: dummy_tool_function for tool in toolkit.tools.values()}
    start_time = time.time()
    result = await graph.execute(tool_functions, toolkit.to_json_schema(), {}, tool_maker)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Time taken to execute complex query: {execution_time:.4f} seconds")
    
    assert len(graph.sub_queries) >= chain_length  # There might be additional sub-queries for parallel execution
    assert result.status == "success"
    
    return routing_time, execution_time, len(graph.sub_queries)

# 4.2 Concurrency Testing
@pytest.mark.asyncio
async def test_parallel_vs_sequential_execution():
    toolkit = Toolkit()
    for name, schema in create_dummy_tools(5).items():
        tool = Tool(name=name, schema=schema["parameters"], description=schema["description"])
        toolkit.add_tool(tool)
    
    tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
    router = Router(toolkit, tool_maker)
    
    queries = [f"Use {tool.name}" for tool in toolkit.tools.values()]
    
    async def execute_query(query):
        graph = router.route(query)
        tool_functions = {tool.name: dummy_tool_function for tool in toolkit.tools.values()}
        return await graph.execute(tool_functions, toolkit.to_json_schema(), {}, tool_maker)
    
    # Parallel execution
    start_time = time.time()
    parallel_results = await asyncio.gather(*[execute_query(query) for query in queries])
    parallel_time = time.time() - start_time
    
    # Sequential execution
    start_time = time.time()
    sequential_results = []
    for query in queries:
        result = await execute_query(query)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Time taken for parallel execution: {parallel_time} seconds")
    print(f"Time taken for sequential execution: {sequential_time} seconds")
    
    assert all(result.status == "success" for result in parallel_results)
    assert all(result.status == "success" for result in sequential_results)
    assert parallel_time < sequential_time


if __name__ == "__main__":
    pytest.main([__file__])
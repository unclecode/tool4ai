# tests/test_toolkit.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
from tool4ai.core.tool import Tool
from tool4ai.core.toolkit import Toolkit

@pytest.fixture
def sample_tool():
    return Tool(
        name="sample_tool",
        schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        description="A sample tool"
    )

def test_toolkit_add_tool(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    assert toolkit.has_tool(sample_tool.id)
    assert toolkit.has_tool(sample_tool.name)

def test_toolkit_remove_tool(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    toolkit.remove_tool(sample_tool.id)
    assert not toolkit.has_tool(sample_tool.id)

def test_toolkit_get_tool(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    retrieved_tool = toolkit.get_tool(sample_tool.id)
    assert retrieved_tool.name == sample_tool.name

def test_toolkit_list_tools(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    tools_list = toolkit.list_tools()
    assert len(tools_list) == 1
    assert tools_list[0].name == sample_tool.name

def test_toolkit_to_json_schema(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    json_schema = toolkit.to_json_schema()
    assert sample_tool.id in json_schema
    assert json_schema[sample_tool.id]["name"] == sample_tool.name

def test_toolkit_to_markdown(sample_tool):
    toolkit = Toolkit()
    toolkit.add_tool(sample_tool)
    markdown = toolkit.to_markdown()
    assert sample_tool.name in markdown
    assert sample_tool.description in markdown
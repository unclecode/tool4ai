# tests/test_tool.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
from tool4ai.core.tool import Tool

def test_tool_creation():
    tool = Tool(
        name="test_tool",
        schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        description="A test tool"
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert "properties" in tool.schema

def test_tool_to_json_schema():
    tool = Tool(
        name="test_tool",
        schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        description="A test tool"
    )
    json_schema = tool.to_json_schema()
    assert json_schema["name"] == "test_tool"
    assert json_schema["description"] == "A test tool"
    assert "schema" in json_schema

def test_tool_from_json_schema():
    json_schema = {
        "id": "123",
        "name": "test_tool",
        "schema": {"type": "object", "properties": {"arg1": {"type": "string"}}},
        "description": "A test tool"
    }
    tool = Tool.from_json_schema(json_schema)
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.id == "123"
    assert "properties" in tool.schema
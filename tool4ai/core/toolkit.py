# File: tool4ai/core/toolkit.py

from typing import Dict, List, Optional
from .tool import Tool

class Toolkit:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.id_to_name: Dict[str, str] = {}
        self.name_to_id: Dict[str, str] = {}

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.id] = tool
        self.id_to_name[tool.id] = tool.name
        self.name_to_id[tool.name] = tool.id

    def remove_tool(self, tool_id: str) -> None:
        tool = self.tools.get(tool_id)
        if tool.id in self.tools:
            del self.tools[tool_id]

    def get_tool(self, tool_id_or_name: str) -> Optional[Tool]:
        tool = self.tools.get(tool_id_or_name)
        if tool:
            return tool
        tool_id = self.name_to_id.get(tool_id_or_name)
        if tool_id:
            return self.tools.get(tool_id)
        return None

    def list_tools(self) -> List[Tool]:
        return list(self.tools.values())

    def to_json_schema(self) -> Dict[str, Dict]:
        return {tool_id: tool.to_json_schema() for tool_id, tool in self.tools.items()}
    
    def has_tool(self, tool_id_or_name: str) -> bool:
        return tool_id_or_name in self.tools or tool_id_or_name in self.name_to_id
    
    def to_markdown(self) -> str:
        return "\n\n".join([self._format_tool_from_schema(tool) for tool_name, tool in self.tools.items()])
    
    def _format_tool_from_schema(self, tool : Tool) -> str:
        tool_schema = tool.to_json_schema()
        tool_name = tool.name
        # Extract the tool description
        description = tool_schema.get("description", "")
        
        # Start with the tool name and description
        formatted_tool = f"{tool_name}: {description}\n"
        
        # Extract the parameters
        parameters = tool_schema.get("schema", {}).get("properties", {})
        
        for param_name, param_details in parameters.items():
            param_type = param_details.get("type", "")
            param_description = param_details.get("description", "")
            
            # Handle arrays differently to match the expected output format
            if param_type == "array":
                item_type = param_details.get("items", {}).get("type", "str")
                formatted_tool += f"    - {param_name} (List[{item_type}]): {param_description}\n"
            else:
                formatted_tool += f"    - {param_name} ({param_type}): {param_description}\n"
        
        return formatted_tool

    @classmethod
    def from_json_schema(cls, data: Dict[str, Dict]) -> 'Toolkit':
        toolkit = cls()
        for tool_data in data.values():
            toolkit.add_tool(Tool.from_json_schema(tool_data))
        return toolkit

    def __repr__(self) -> str:
        return f"Toolkit(tools={list(self.tools.keys())})"
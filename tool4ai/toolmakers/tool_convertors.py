# File: tool4ai/strategies/llm_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ToolsConvertor(ABC):
    """
    Abstract base class for tools convertor.
    """

    @abstractmethod
    def convert(self, tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

  
class OpenAIToolConvertor(ToolsConvertor):
    """
    A tool convertor for OpenAI tools.
    """

    def convert(self, tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted_tools = []
        
        for tool in tools_info:
            tool["schema"].setdefault("additionalProperties", False)
            converted_tool = {
                "name": tool["name"],
                "strict": True,
                "description": tool["description"],
                "parameters": tool["schema"] #{**tool["schema"], "additionalProperties" : False}
            }
            converted_tools.append({
                "type": "function",
                "function": converted_tool
            })
        return converted_tools  

class AnthropicToolConvertor(ToolsConvertor):
    """
    A tool convertor for Anthropic tools.
    """

    def convert(self, tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted_tools = []
        for tool in tools_info:
            converted_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["schema"],
            }
            converted_tools.append(converted_tool)
        return converted_tools    
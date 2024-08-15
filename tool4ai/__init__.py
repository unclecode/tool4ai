# File: tool4ai/__init__.py

"""
Tool4AI: A model agnostic and LLM friendly router for tool/function call.

This library provides functionality for managing tools, creating toolkits,
and routing decisions based on LLM outputs for effective tool/function calls.
"""

from .core.tool import Tool
from .core.toolkit import Toolkit
from .core.router import Router
from .core.models import SubQuery, SubQueryResponse
from .utils.dependency_graph import DependencyGraph
from .utils.tool_dependency_graph import ToolDependencyGraph
from .utils.config_manager import config_manager
from .toolmakers.llm_strategy import LLMStrategy, NativeStrategy, AnthropicStrategy

# Version of the tool4ai package
__version__ = "0.1.0"

# Define what should be importable from the package
__all__ = ['Tool', 'Toolkit', 'Router', 'DependencyGraph', 'config_manager', 'LLMStrategy', 'ToolDependencyGraph', 'SubQuery', 'SubQueryResponse', 'NativeStrategy', 'AnthropicStrategy']

# Package level initialization code (if any)
def initialize():
    """
    Perform any necessary package-level initialization.
    This function is called when the package is imported.
    """
    # For now, we don't need any initialization, but we can add code here if needed in the future
    pass

# Call the initialize function
initialize()
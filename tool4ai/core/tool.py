# File: tool4ai/core/tool.py

from typing import Union, Dict, Any
from pydantic import BaseModel
import json, uuid

class Tool:
    """
    Represents a single tool or function that can be called by the AI.
    """

    def __init__(self, name: str, schema: Union[str, Dict[str, Any]], description: str):
        self.id : str = str(uuid.uuid4()) 
        self.name : str = name
        self.schema : Dict[str, Any] = schema if isinstance(schema, Dict) else json.loads(schema)
        self.description : str = description

    
    def to_json_schema(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schema": self.schema,
            "description": self.description
        }

    @classmethod
    def from_json_schema(cls, data: Dict[str, Any]) -> 'Tool':
        tool = cls(
            name=data["name"],
            schema=data["schema"],
            description=data["description"]
        )
        tool.id = data["id"] or str(uuid.uuid4())
        return tool

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')"
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class SubQuery(BaseModel):
    index: int = 0
    sub_query: str
    task: str
    tool: Optional[str] = None
    dependent_on: int = -1
    dependency_attr: Optional[str] = ""
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tool_missing: Optional[bool] = False
    result: Optional[str] = None 
    status: Optional[str] = "pending"
    issue: Optional[str] = None
    actionable: Optional[bool] = True
    is_orphan: Optional[bool] = False

class SubQueryResponse(BaseModel):
    sub_queries: List[SubQuery]

class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    HUMAN = "human"
    ERROR = "error"
    PENDING = "pending"
    NEW_DISCUSSION = "new_discussion"

class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra='ignore')
    status: ExecutionStatus
    message: str
    memory: List[Dict[str, Any]]
    sub_queries: List[SubQuery]
    error_info: Optional[Dict[str, Any]] = None
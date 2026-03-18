from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class Parameter(BaseModel):
    """An input parameter for an endpoint."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str
    required: bool
    description: str = ""
    enum_values: list[str] = []
    default: Any = None


class Endpoint(BaseModel):
    """A callable action within a tool."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    method: str = "GET"
    parameters: list[Parameter] = []
    response_fields: list[str] = []


class Tool(BaseModel):
    """A top-level API tool from ToolBench."""

    model_config = ConfigDict(extra="ignore")

    tool_id: str
    name: str
    description: str = ""
    category: str = "general"
    endpoints: list[Endpoint] = []
    source_data: dict = {}

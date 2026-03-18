from __future__ import annotations

from enum import Enum


class NodeType(Enum):
    """Types of nodes in the Tool Graph."""

    TOOL = "tool"
    ENDPOINT = "endpoint"
    PARAMETER = "parameter"
    RESPONSE_FIELD = "response_field"
    CONCEPT = "concept"


class EdgeType(Enum):
    """Types of edges in the Tool Graph."""

    HAS_ENDPOINT = "has_endpoint"
    HAS_PARAMETER = "has_parameter"
    RETURNS = "returns"
    TAGGED_WITH = "tagged_with"
    COMPATIBLE_WITH = "compatible_with"


def node_id(node_type: NodeType, name: str, parent: str | None = None) -> str:
    """Return a consistent node ID string.

    Args:
        node_type: The type of node
        name: The node name
        parent: Optional parent context to avoid collisions

    Examples:
        node_id(NodeType.TOOL, "weather_api")
          → "tool:weather_api"
        node_id(NodeType.ENDPOINT, "get_forecast", parent="weather_api")
          → "endpoint:weather_api.get_forecast"
        node_id(NodeType.PARAMETER, "city", parent="weather_api.get_forecast")
          → "parameter:weather_api.get_forecast.city"
        node_id(NodeType.CONCEPT, "travel")
          → "concept:travel"
    """
    if parent:
        return f"{node_type.value}:{parent}.{name}"
    return f"{node_type.value}:{name}"

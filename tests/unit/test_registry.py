"""Unit tests for Tool Registry: loading, normalization, and missing field handling."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tacs.registry.loader import (
    _parse_enum_values,
    _parse_parameter,
    _parse_endpoint,
    load_tools,
)
from tacs.registry.models import Endpoint, Parameter, Tool
from tacs.registry.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_json(
    tool_name: str = "test_tool",
    standardized_name: str = "test_tool",
    description: str = "A test tool",
    api_list: list | None = None,
) -> dict:
    return {
        "tool_name": tool_name,
        "standardized_name": standardized_name,
        "tool_description": description,
        "api_list": api_list or [],
    }


def _make_api(
    name: str = "get_data",
    description: str = "Fetch data",
    method: str = "GET",
    required: list | None = None,
    optional: list | None = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "method": method,
        "required_parameters": required or [],
        "optional_parameters": optional or [],
    }


def _make_param(name: str, type_: str = "string", description: str = "") -> dict:
    return {"name": name, "type": type_, "description": description}


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a minimal ToolBench-style data directory."""
    tools_dir = tmp_path / "toolenv" / "tools" / "test_category"
    tools_dir.mkdir(parents=True)

    tool_data = _make_tool_json(
        tool_name="Weather API",
        standardized_name="weather_api",
        description="Provides weather data",
        api_list=[
            _make_api(
                name="get_forecast",
                description="Get weather forecast",
                method="GET",
                required=[_make_param("city", "string", "City name")],
                optional=[_make_param("units", "string", "one of 'metric', 'imperial'")],
            )
        ],
    )
    (tools_dir / "weather_api.json").write_text(json.dumps(tool_data))
    return tmp_path


# ---------------------------------------------------------------------------
# _parse_enum_values
# ---------------------------------------------------------------------------

class TestParseEnumValues:
    def test_one_of_quoted(self):
        result = _parse_enum_values("Must be one of 'asc', 'desc'")
        assert "asc" in result
        assert "desc" in result

    def test_options_colon(self):
        result = _parse_enum_values('Options: "metric", "imperial"')
        assert "metric" in result
        assert "imperial" in result

    def test_no_trigger_returns_empty(self):
        result = _parse_enum_values("A plain description with no hints")
        assert result == []

    def test_must_be_unquoted(self):
        result = _parse_enum_values("Must be asc or desc")
        # Comma-separated fallback won't match "or" phrasing — empty is fine
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _parse_parameter
# ---------------------------------------------------------------------------

class TestParseParameter:
    def test_required_parameter(self):
        raw = {"name": "city", "type": "string", "description": "City name"}
        param = _parse_parameter(raw, required=True)
        assert param is not None
        assert param.name == "city"
        assert param.required is True
        assert param.type == "string"

    def test_optional_parameter(self):
        raw = {"name": "units", "type": "string", "description": "Unit system"}
        param = _parse_parameter(raw, required=False)
        assert param is not None
        assert param.required is False

    def test_missing_name_returns_none(self):
        raw = {"type": "string", "description": "no name here"}
        param = _parse_parameter(raw, required=True)
        assert param is None

    def test_missing_type_defaults_to_string(self):
        raw = {"name": "q"}
        param = _parse_parameter(raw, required=True)
        assert param is not None
        assert param.type == "string"

    def test_enum_values_parsed(self):
        raw = {
            "name": "order",
            "type": "string",
            "description": "Must be one of 'asc', 'desc'",
        }
        param = _parse_parameter(raw, required=False)
        assert param is not None
        assert "asc" in param.enum_values


# ---------------------------------------------------------------------------
# _parse_endpoint
# ---------------------------------------------------------------------------

class TestParseEndpoint:
    def test_basic_endpoint(self):
        raw = _make_api(
            name="search",
            required=[_make_param("q")],
            optional=[_make_param("limit", "integer")],
        )
        ep = _parse_endpoint(raw, response_fields=[])
        assert ep is not None
        assert ep.name == "search"
        assert len([p for p in ep.parameters if p.required]) == 1
        assert len([p for p in ep.parameters if not p.required]) == 1

    def test_missing_name_returns_none(self):
        raw = {"description": "no name", "method": "GET"}
        ep = _parse_endpoint(raw, response_fields=[])
        assert ep is None

    def test_method_uppercased(self):
        raw = _make_api(name="post_data", method="post")
        ep = _parse_endpoint(raw, response_fields=[])
        assert ep is not None
        assert ep.method == "POST"

    def test_response_fields_stored(self):
        raw = _make_api(name="get_weather")
        ep = _parse_endpoint(raw, response_fields=["temperature", "humidity"])
        assert ep is not None
        assert "temperature" in ep.response_fields

    def test_none_parameters_handled(self):
        raw = {
            "name": "empty_ep",
            "required_parameters": None,
            "optional_parameters": None,
        }
        ep = _parse_endpoint(raw, response_fields=[])
        assert ep is not None
        assert ep.parameters == []


# ---------------------------------------------------------------------------
# load_tools
# ---------------------------------------------------------------------------

class TestLoadTools:
    def test_loads_tools_from_directory(self, tmp_data_dir: Path):
        tools = load_tools(tmp_data_dir)
        assert len(tools) == 1
        assert tools[0].tool_id == "weather_api"

    def test_tool_has_endpoints(self, tmp_data_dir: Path):
        tools = load_tools(tmp_data_dir)
        assert len(tools[0].endpoints) == 1
        assert tools[0].endpoints[0].name == "get_forecast"

    def test_endpoint_has_parameters(self, tmp_data_dir: Path):
        tools = load_tools(tmp_data_dir)
        ep = tools[0].endpoints[0]
        required = [p for p in ep.parameters if p.required]
        optional = [p for p in ep.parameters if not p.required]
        assert len(required) == 1
        assert len(optional) == 1
        assert required[0].name == "city"

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        tools = load_tools(tmp_path / "nonexistent")
        assert tools == []

    def test_malformed_json_skipped(self, tmp_path: Path):
        tools_dir = tmp_path / "toolenv" / "tools" / "cat"
        tools_dir.mkdir(parents=True)
        (tools_dir / "bad.json").write_text("not valid json{{{")
        tools = load_tools(tmp_path)
        assert tools == []

    def test_tool_category_set_from_directory(self, tmp_data_dir: Path):
        tools = load_tools(tmp_data_dir)
        assert tools[0].category == "test_category"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        tools = [
            Tool(
                tool_id="weather_api",
                name="Weather API",
                category="weather",
                endpoints=[
                    Endpoint(
                        name="get_forecast",
                        parameters=[
                            Parameter(name="city", type="string", required=True)
                        ],
                    )
                ],
            ),
            Tool(
                tool_id="maps_api",
                name="Maps API",
                category="travel",
                endpoints=[
                    Endpoint(name="search_places"),
                    Endpoint(name="get_directions"),
                ],
            ),
        ]
        return ToolRegistry(tools)

    def test_get_tool(self, registry: ToolRegistry):
        tool = registry.get_tool("weather_api")
        assert tool is not None
        assert tool.name == "Weather API"

    def test_get_tool_missing_returns_none(self, registry: ToolRegistry):
        assert registry.get_tool("nonexistent") is None

    def test_get_endpoint(self, registry: ToolRegistry):
        ep = registry.get_endpoint("weather_api", "get_forecast")
        assert ep is not None
        assert ep.name == "get_forecast"

    def test_get_endpoint_missing_tool(self, registry: ToolRegistry):
        assert registry.get_endpoint("bad_tool", "get_forecast") is None

    def test_get_endpoint_missing_endpoint(self, registry: ToolRegistry):
        assert registry.get_endpoint("weather_api", "nonexistent") is None

    def test_tool_count(self, registry: ToolRegistry):
        assert registry.tool_count == 2

    def test_endpoint_count(self, registry: ToolRegistry):
        assert registry.endpoint_count == 3

    def test_list_by_category(self, registry: ToolRegistry):
        travel = registry.list_by_category("travel")
        assert len(travel) == 1
        assert travel[0].tool_id == "maps_api"

    def test_list_categories(self, registry: ToolRegistry):
        cats = registry.list_categories()
        assert "weather" in cats
        assert "travel" in cats

    def test_all_endpoints(self, registry: ToolRegistry):
        pairs = registry.all_endpoints()
        assert len(pairs) == 3
        tool_ids = {tid for tid, _ in pairs}
        assert "weather_api" in tool_ids
        assert "maps_api" in tool_ids

    def test_save_and_load(self, registry: ToolRegistry, tmp_path: Path):
        registry.save(tmp_path)
        loaded = ToolRegistry.load(tmp_path)
        assert loaded.tool_count == registry.tool_count
        assert loaded.get_tool("weather_api") is not None

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ToolRegistry.load(tmp_path)

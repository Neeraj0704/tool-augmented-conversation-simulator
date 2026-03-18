from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tacs.registry.models import Endpoint, Parameter, Tool

logger = logging.getLogger(__name__)


def _parse_enum_values(description: str) -> list[str]:
    """Try to extract enum values from a parameter description string."""
    desc_lower = description.lower()
    if not any(kw in desc_lower for kw in ("one of", "options:", "must be")):
        return []

    # Match quoted values: "foo", 'bar'
    quoted = re.findall(r'["\']([^"\']+)["\']', description)
    if quoted:
        return quoted

    # Match comma-separated words after trigger phrases
    match = re.search(
        r'(?:one of|options:|must be)[:\s]+([a-zA-Z0-9_, ]+)',
        description,
        re.IGNORECASE,
    )
    if match:
        values = [v.strip() for v in match.group(1).split(",") if v.strip()]
        if values:
            return values

    return []


def _parse_parameter(raw: dict, required: bool) -> Parameter | None:
    """Parse a single raw parameter dict into a Parameter model."""
    try:
        name = raw.get("name", "").strip()
        if not name:
            return None
        description = raw.get("description", "") or ""
        return Parameter(
            name=name,
            type=raw.get("type", "string") or "string",
            required=required,
            description=description,
            enum_values=_parse_enum_values(description),
            default=raw.get("default", None),
        )
    except Exception as exc:
        logger.warning("Failed to parse parameter %s: %s", raw.get("name"), exc)
        return None


def _parse_endpoint(raw: dict, response_fields: list[str]) -> Endpoint | None:
    """Parse a single raw api_list entry into an Endpoint model."""
    try:
        name = raw.get("name", "").strip()
        if not name:
            return None

        parameters: list[Parameter] = []
        for p in raw.get("required_parameters", []) or []:
            param = _parse_parameter(p, required=True)
            if param:
                parameters.append(param)
        for p in raw.get("optional_parameters", []) or []:
            param = _parse_parameter(p, required=False)
            if param:
                parameters.append(param)

        return Endpoint(
            name=name,
            description=raw.get("description", "") or "",
            method=(raw.get("method", "GET") or "GET").upper(),
            parameters=parameters,
            response_fields=response_fields,
        )
    except Exception as exc:
        logger.warning("Failed to parse endpoint %s: %s", raw.get("name"), exc)
        return None


def _load_response_fields(
    response_dir: Path, tool_id: str
) -> dict[str, list[str]]:
    """
    Load response field keys per endpoint from response_examples directory.
    Returns a dict mapping endpoint name -> list of top-level response keys.
    """
    result: dict[str, list[str]] = {}
    tool_response_dir = response_dir / tool_id
    if not tool_response_dir.exists():
        return result

    for example_file in tool_response_dir.glob("*.json"):
        try:
            data = json.loads(example_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                endpoint_name = example_file.stem
                result[endpoint_name] = list(data.keys())
        except Exception as exc:
            logger.warning(
                "Failed to read response example %s: %s", example_file, exc
            )
    return result


def _load_tool(json_path: Path, category: str, response_dir: Path) -> Tool | None:
    """Parse a single ToolBench JSON file into a Tool model."""
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Malformed JSON, skipping %s: %s", json_path, exc)
        return None

    try:
        # Tool identity
        tool_name = raw.get("tool_name") or raw.get("name") or json_path.stem
        standardized = raw.get("standardized_name", "")
        tool_id = standardized if standardized else tool_name.lower().replace(" ", "_")

        # Description
        description = (
            raw.get("tool_description")
            or raw.get("description")
            or ""
        )

        # Endpoints — handle both api_list and apis keys
        api_list = raw.get("api_list") or raw.get("apis") or []
        response_fields_map = _load_response_fields(response_dir, tool_id)

        endpoints: list[Endpoint] = []
        for api_raw in api_list:
            endpoint = _parse_endpoint(
                api_raw,
                response_fields=response_fields_map.get(api_raw.get("name", ""), []),
            )
            if endpoint:
                endpoints.append(endpoint)

        return Tool(
            tool_id=tool_id,
            name=tool_name,
            description=description,
            category=category,
            endpoints=endpoints,
            source_data=raw,
        )
    except Exception as exc:
        logger.warning("Failed to build Tool from %s: %s", json_path, exc)
        return None


def load_tools(data_dir: Path) -> list[Tool]:
    """
    Walk data_dir/toolenv/tools/, parse every tool JSON, and return
    a list of Tool models. Malformed files are logged and skipped.
    """
    tools_dir = data_dir / "toolenv" / "tools"
    response_dir = data_dir / "toolenv" / "response_examples"

    if not tools_dir.exists():
        logger.error("Tools directory not found: %s", tools_dir)
        return []

    tools: list[Tool] = []
    for category_dir in sorted(tools_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for json_path in sorted(category_dir.glob("*.json")):
            tool = _load_tool(json_path, category, response_dir)
            if tool:
                tools.append(tool)

    logger.info("Loaded %d tools from %s", len(tools), tools_dir)
    return tools

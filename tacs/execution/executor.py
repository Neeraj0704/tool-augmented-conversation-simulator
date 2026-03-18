from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any

from pydantic import BaseModel, ConfigDict

from tacs.registry.registry import ToolRegistry

logger = logging.getLogger(__name__)

_REALISTIC_NAMES = [
    "Alice Johnson", "Bob Smith", "Carol White", "David Lee",
    "Emma Davis", "Frank Miller", "Grace Wilson", "Henry Moore",
]


class MockResult(BaseModel):
    """Result of a mock tool execution."""

    model_config = ConfigDict(extra="ignore")

    output: dict[str, Any]
    valid: bool
    errors: list[str]


class MockExecutor:
    """Validates tool call arguments and returns deterministic mock responses."""

    def __init__(self, registry: ToolRegistry, seed: int) -> None:
        self._registry = registry
        self._seed = seed

    def execute(
        self,
        tool_id: str,
        endpoint_name: str,
        arguments: dict[str, Any],
    ) -> MockResult:
        """Validate arguments and return a deterministic mock response."""
        # Step 1 — Validation
        endpoint = self._registry.get_endpoint(tool_id, endpoint_name)
        if endpoint is None:
            logger.warning(
                "Endpoint not found: tool_id=%s endpoint=%s", tool_id, endpoint_name
            )
            return MockResult(output={}, valid=False, errors=["Endpoint not found"])

        errors: list[str] = []
        for param in endpoint.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")

        valid = len(errors) == 0
        if not valid:
            logger.warning(
                "Validation failed: tool_id=%s endpoint=%s errors=%s",
                tool_id, endpoint_name, errors,
            )

        # Step 2 — Deterministic RNG from hash of inputs
        hash_input = f"{self._seed}:{tool_id}:{endpoint_name}:{json.dumps(arguments, sort_keys=True)}"
        hash_int = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        rng = random.Random(hash_int)

        # Step 3 — Generate mock response fields
        output: dict[str, Any] = {}
        fields = endpoint.response_fields or []

        if fields:
            for field in fields:
                output[field] = self._mock_value(field, endpoint_name, rng)
        else:
            # Generic response when no schema available
            output["result"] = f"{endpoint_name}_{rng.randint(1, 999)}"
            output["success"] = True

        # Step 4 — Echo arguments for chain consistency
        output["_input"] = arguments

        logger.debug(
            "MockExecutor: tool_id=%s endpoint=%s valid=%s",
            tool_id, endpoint_name, valid,
        )
        return MockResult(output=output, valid=valid, errors=errors)

    def _mock_value(self, field: str, endpoint_name: str, rng: random.Random) -> Any:
        """Generate a realistic mock value based on field name."""
        f = field.lower()
        if "id" in f:
            return f"{endpoint_name}_{rng.randint(100, 999)}"
        elif "name" in f:
            return rng.choice(_REALISTIC_NAMES)
        elif "price" in f or "cost" in f:
            return rng.randint(10, 2000)
        elif "date" in f:
            return f"2024-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"
        elif "url" in f:
            return f"https://example.com/{endpoint_name}"
        elif "count" in f or "total" in f:
            return rng.randint(1, 100)
        elif "status" in f:
            return rng.choice(["active", "pending", "confirmed"])
        else:
            return f"{field}_{rng.randint(1, 999)}"

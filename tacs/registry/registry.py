from __future__ import annotations

import logging
import pickle
from pathlib import Path

from tacs.registry.models import Endpoint, Tool

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.pkl"


class ToolRegistry:
    """In-memory registry of all loaded Tool models with lookup methods."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.tool_id: t for t in tools}
        logger.info("ToolRegistry initialised with %d tools", len(self._tools))

    # --- Lookups ---

    def get_tool(self, tool_id: str) -> Tool | None:
        """Return a Tool by its tool_id, or None if not found."""
        return self._tools.get(tool_id)

    def get_endpoint(self, tool_id: str, endpoint_name: str) -> Endpoint | None:
        """Return a specific Endpoint from a Tool, or None if not found."""
        tool = self._tools.get(tool_id)
        if not tool:
            return None
        for endpoint in tool.endpoints:
            if endpoint.name == endpoint_name:
                return endpoint
        return None

    def list_tools(self) -> list[Tool]:
        """Return all tools."""
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[Tool]:
        """Return all tools belonging to a given category."""
        return [t for t in self._tools.values() if t.category == category]

    def list_categories(self) -> list[str]:
        """Return sorted list of all unique categories."""
        return sorted({t.category for t in self._tools.values()})

    def all_endpoints(self) -> list[tuple[str, Endpoint]]:
        """Return all (tool_id, Endpoint) pairs across the registry."""
        return [
            (tool_id, ep)
            for tool_id, tool in self._tools.items()
            for ep in tool.endpoints
        ]

    # --- Stats ---

    @property
    def tool_count(self) -> int:
        """Total number of tools in the registry."""
        return len(self._tools)

    @property
    def endpoint_count(self) -> int:
        """Total number of endpoints across all tools."""
        return sum(len(t.endpoints) for t in self._tools.values())

    @property
    def category_count(self) -> int:
        """Total number of unique categories."""
        return len({t.category for t in self._tools.values()})

    # --- Persistence ---

    def save(self, artifacts_dir: Path) -> None:
        """Pickle the registry to artifacts_dir/registry.pkl."""
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = artifacts_dir / _REGISTRY_FILE
            with open(out_path, "wb") as f:
                pickle.dump(self, f)
            logger.info("Saved registry (%d tools) to %s", self.tool_count, out_path)
        except OSError:
            logger.exception("Failed to save registry to %s", artifacts_dir)
            raise

    @classmethod
    def load(cls, artifacts_dir: Path) -> ToolRegistry:
        """Load a pickled registry from artifacts_dir/registry.pkl."""
        pkl_path = artifacts_dir / _REGISTRY_FILE
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Registry artifact not found at {pkl_path}. Run `tacs build` first."
            )
        try:
            with open(pkl_path, "rb") as f:
                registry = pickle.load(f)
            logger.info("Loaded registry (%d tools) from %s", registry.tool_count, pkl_path)
            return registry
        except (OSError, pickle.UnpicklingError, EOFError):
            logger.exception("Failed to load registry from %s", pkl_path)
            raise

    def __len__(self) -> int:
        return len(self._tools)

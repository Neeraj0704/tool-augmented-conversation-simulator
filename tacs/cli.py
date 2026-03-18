from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import click

from tacs.config import config

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


@click.group()
def cli() -> None:
    """Tool Augmented Conversation Simulator (TACS)."""
    _setup_logging()


# ---------------------------------------------------------------------------
# tacs build
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: config.data_dir,
    show_default=True,
    help="Directory containing ToolBench raw data.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: config.artifacts_dir,
    show_default=True,
    help="Directory to write registry.pkl, graph.pkl, and build_meta.json.",
)
def build(data_dir: Path, output_dir: Path) -> None:
    """Build the tool registry and graph from ToolBench data."""
    from tacs.graph.builder import ToolGraphBuilder
    from tacs.registry.loader import load_tools
    from tacs.registry.registry import ToolRegistry

    click.echo(f"Loading tools from {data_dir} …")
    tools = load_tools(data_dir)
    if not tools:
        raise click.ClickException(
            f"No tools found in {data_dir}. "
            "Check that data/toolenv/tools/ exists and is populated."
        )

    registry = ToolRegistry(tools)
    registry.save(output_dir)

    click.echo("Building tool graph …")
    builder = ToolGraphBuilder(registry)
    graph = builder.build()
    ToolGraphBuilder.save(graph, output_dir)

    meta = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "tool_count": registry.tool_count,
        "endpoint_count": registry.endpoint_count,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }
    meta_path = output_dir / "build_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    click.echo(
        f"Build complete: {registry.tool_count} tools, "
        f"{registry.endpoint_count} endpoints, "
        f"{graph.number_of_nodes()} graph nodes → {output_dir}"
    )


# ---------------------------------------------------------------------------
# tacs generate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--seed", type=int, default=lambda: config.seed, show_default=True)
@click.option(
    "--count",
    type=int,
    default=lambda: config.conversation_count,
    show_default=True,
    help="Number of conversations to generate.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=lambda: config.output_dir / "conversations.jsonl",
    show_default=True,
)
@click.option(
    "--artifacts-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: config.artifacts_dir,
    show_default=True,
)
@click.option(
    "--corpus-memory/--no-corpus-memory",
    default=True,
    show_default=True,
    help="Enable corpus memory (Run B) or disable (Run A).",
)
def generate(
    seed: int,
    count: int,
    output: Path,
    artifacts_dir: Path,
    corpus_memory: bool,
) -> None:
    """Generate synthetic conversations using the built registry and graph."""
    from tacs.agents.pipeline import generate as run_generate
    from tacs.config import Config
    from tacs.graph.builder import ToolGraphBuilder
    from tacs.llm import LLMClient
    from tacs.memory.store import MemoryStore
    from tacs.registry.registry import ToolRegistry

    cfg = Config().model_copy(update={"seed": seed})

    registry = ToolRegistry.load(artifacts_dir)
    graph = ToolGraphBuilder.load(artifacts_dir)
    llm = LLMClient(backend=cfg.llm_backend)
    memory = MemoryStore()

    click.echo(
        f"Generating {count} conversations "
        f"(seed={seed}, corpus_memory={corpus_memory}) → {output}"
    )
    run_generate(
        registry=registry,
        graph=graph,
        llm=llm,
        config=cfg,
        memory=memory,
        count=count,
        corpus_memory_enabled=corpus_memory,
        output_path=output,
    )
    click.echo("Done.")


# ---------------------------------------------------------------------------
# tacs validate
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file produced by tacs generate.",
)
def validate(input_path: Path) -> None:
    """Validate every conversation in a JSONL file and report issues."""
    from tacs.agents.models import Conversation
    from tacs.agents.validator_agent import ValidatorAgent
    from tacs.llm import LLMClient

    llm = LLMClient(backend=config.llm_backend)
    validator = ValidatorAgent(llm, config)

    total = 0
    invalid = 0
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            conv = Conversation.model_validate_json(line)
        except Exception as exc:
            invalid += 1
            logger.warning("Line %d JSON parse error: %s", total, exc)
            continue

        result = validator.run(conv)
        if not result.valid:
            invalid += 1
            logger.warning(
                "conv %s failed: %s",
                conv.conversation_id,
                result.errors,
            )

    click.echo(f"Validated {total} conversations: {total - invalid} ok, {invalid} invalid.")
    if invalid:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# tacs metrics
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Single JSONL file to compute metrics for.",
)
@click.option(
    "--input-a",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Run A JSONL (corpus memory disabled).",
)
@click.option(
    "--input-b",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Run B JSONL (corpus memory enabled).",
)
def metrics(
    input_path: Path | None,
    input_a: Path | None,
    input_b: Path | None,
) -> None:
    """Compute quality and diversity metrics for generated conversations."""
    from tacs.agents.models import Conversation

    def _load(path: Path) -> list[Conversation]:
        convs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                convs.append(Conversation.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed line in %s: %s", path, exc)
        return convs

    def _report(convs: list[Conversation], label: str) -> None:
        if not convs:
            click.echo(f"{label}: no conversations.")
            return

        n = len(convs)
        avg_turns = sum(c.metadata.num_turns for c in convs) / n
        avg_tool_calls = sum(len(c.tool_calls) for c in convs) / n
        avg_clarifications = (
            sum(c.metadata.num_clarification_questions for c in convs) / n
        )
        grounding_rates = [
            c.metadata.memory_grounding_rate
            for c in convs
            if c.metadata.memory_grounding_rate is not None
        ]
        avg_grounding = (
            sum(grounding_rates) / len(grounding_rates) if grounding_rates else None
        )
        multi_step = sum(1 for c in convs if len(c.tool_calls) >= 3) / n
        multi_tool = (
            sum(
                1
                for c in convs
                if len(set(c.metadata.tool_ids_used)) >= 2
            )
            / n
        )

        click.echo(f"\n── {label} ({n} conversations) ──")
        click.echo(f"  avg turns:               {avg_turns:.2f}")
        click.echo(f"  avg tool calls:          {avg_tool_calls:.2f}")
        click.echo(f"  avg clarifications:      {avg_clarifications:.2f}")
        click.echo(
            f"  avg memory grounding:    "
            f"{avg_grounding:.2f}" if avg_grounding is not None else "  avg memory grounding:    n/a"
        )
        click.echo(f"  multi-step (≥3 calls):   {multi_step:.1%}")
        click.echo(f"  multi-tool (≥2 tools):   {multi_tool:.1%}")

    def _jaccard_diversity(convs: list[Conversation]) -> float:
        """Mean pairwise Jaccard dissimilarity over tool-chain sets."""
        chains = [set(c.metadata.tool_ids_used) for c in convs]
        if len(chains) < 2:
            return 0.0
        total, pairs = 0.0, 0
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                a, b = chains[i], chains[j]
                union = len(a | b)
                inter = len(a & b)
                total += 1.0 - (inter / union if union else 0.0)
                pairs += 1
        return total / pairs if pairs else 0.0

    if input_path:
        convs = _load(input_path)
        _report(convs, str(input_path))
        click.echo(f"  jaccard diversity:       {_jaccard_diversity(convs):.4f}")

    if input_a and input_b:
        convs_a = _load(input_a)
        convs_b = _load(input_b)
        _report(convs_a, f"Run A — {input_a.name}")
        _report(convs_b, f"Run B — {input_b.name}")
        div_a = _jaccard_diversity(convs_a)
        div_b = _jaccard_diversity(convs_b)
        click.echo(f"\n── Diversity comparison ──")
        click.echo(f"  Run A jaccard diversity: {div_a:.4f}")
        click.echo(f"  Run B jaccard diversity: {div_b:.4f}")
        click.echo(
            f"  Δ (B − A):               {div_b - div_a:+.4f} "
            f"({'more' if div_b > div_a else 'less'} diverse with corpus memory)"
        )

# CLAUDE.md — Project Context for Claude Code

> This file is automatically read by Claude Code. It provides full context
> about the project so every suggestion is aligned with the architecture,
> conventions, and goals.

---

## What This Project Is

**Tool Augmented Conversation Simulator (TACS)** is a synthetic data generation
system that produces multi-turn, multi-tool conversation traces for training and
evaluating tool-use AI agents.

Built as an assessment for SAP's AI Scientist Intern role.

The system:
1. Ingests ToolBench API definitions and normalizes them into a typed Tool Registry
2. Builds a Tool Graph (knowledge graph) over tools, endpoints, parameters, and concepts
3. Samples realistic tool chains from the graph
4. Uses a multi-agent pipeline to generate synthetic conversations
5. Grounds generation in session and corpus memory via mem0
6. Outputs a JSONL dataset with full metadata

---

## Implementation Philosophy

> **Simple + Working + Clean beats Complex + Broken every time.**

This is an assessment submission, not a production platform. The grading rubric is:

| Criterion | Weight |
|---|---|
| Functional correctness | 35% |
| Software engineering / code quality | 35% |
| Knowledge graph + sampling | 10% |
| Multi-agent system design | 10% |
| Memory + diversity analysis | 10% |

**70% of the grade is correctness + clean code. Never sacrifice a working pipeline for clever architecture.**

### What "clean engineering" means here
- ✅ Single-responsibility modules — each file does one thing
- ✅ Pydantic models at module boundaries — not for every internal helper
- ✅ Structured logging instead of print()
- ✅ One central config file — no hardcoded values anywhere
- ✅ Graceful error handling — log and skip bad data, never crash the pipeline
- ✅ Deterministic output — seed flows through everything
- ✅ Docker so it runs on any machine
- ✅ Tests that actually pass

### What to AVOID
- ❌ Over-engineering the graph — clean NetworkX with 5 node types is enough
- ❌ Building multiple sampler variants before the first one works end-to-end
- ❌ Making agents "autonomous" — they are well-named Python classes with focused LLM prompts
- ❌ Pydantic for every internal helper — use it at module boundaries only
- ❌ Any abstraction that risks the pipeline not running end-to-end

### Build Order — Never skip phases
```
Phase 1 — Make it run
  → registry loads ToolBench tools
  → graph builds, sampler proposes a tool chain
  → mock executor produces a fake response
  → one full conversation generates and saves to JSONL

Phase 2 — Make it right
  → session + corpus memory working correctly
  → all 5 agents working with proper prompts
  → 50 conversations generate with all required metadata fields
  → diversity experiment (Run A vs Run B) works

Phase 3 — Make it shine
  → all tests pass
  → Docker runs end to end on a clean machine
  → README + DESIGN.md written clearly
  → logging and error handling cleaned up
```

---

## Repository Layout

```
tool-augmented-conversation-simulator/
├── tacs/                        # Main Python package
│   ├── __init__.py
│   ├── config.py                # Central config (Pydantic BaseSettings)
│   ├── llm.py                   # LLMClient abstraction (openai / anthropic / ollama)
│   ├── cli.py                   # Click CLI: build / generate / validate / metrics
│   ├── registry/                # Part 1 — Tool Registry
│   │   ├── __init__.py
│   │   ├── loader.py            # ToolBench ingestion + normalization
│   │   ├── models.py            # Pydantic models: Tool, Endpoint, Parameter
│   │   └── registry.py         # ToolRegistry class with lookup methods
│   ├── graph/                   # Part 2 — Tool Graph + Sampler
│   │   ├── __init__.py
│   │   ├── builder.py           # Builds NetworkX graph from registry
│   │   ├── models.py            # Node/edge type enums (NodeType, EdgeType)
│   │   └── sampler.py           # ToolChainSampler: multi-step, parallel, hybrid
│   ├── execution/               # Part 3 — Offline Execution Model
│   │   ├── __init__.py
│   │   ├── executor.py          # MockExecutor: validates args, returns mock responses
│   │   └── session.py           # SessionState: tracks outputs within a conversation
│   ├── agents/                  # Part 4 — Multi-Agent System
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAgent abstract class
│   │   ├── models.py            # Pydantic models: Message, ToolCall, Conversation, etc.
│   │   ├── sampler_agent.py     # Proposes tool chains from the graph
│   │   ├── planner_agent.py     # Plans the conversation scenario
│   │   ├── user_proxy.py        # Simulates user utterances
│   │   ├── assistant_agent.py   # Produces tool calls + final responses
│   │   ├── validator_agent.py   # Validates conversation quality
│   │   └── pipeline.py         # Orchestrates all agents into one conversation
│   ├── memory/                  # Part 5 — Agentic Memory
│   │   ├── __init__.py
│   │   └── store.py             # MemoryStore class backed by mem0
│   └── metrics/                 # Evaluation + diversity metrics
│       ├── __init__.py
│       └── evaluator.py         # grounding_rate, jaccard diversity, distinct-N, etc.
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_registry.py     # ToolBench loading + normalization
│   │   ├── test_memory.py       # add→search, scope isolation
│   │   └── test_graph.py        # graph construction + sampling
│   └── e2e/
│       ├── __init__.py
│       └── test_pipeline.py     # build artifacts + generate 50 samples
├── data/                        # ToolBench raw data (downloaded at build, not committed)
├── artifacts/                   # Produced by tacs build — loaded by tacs generate
│   ├── registry.pkl
│   ├── graph.pkl
│   └── build_meta.json
├── output/                      # Generated JSONL conversations
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── DESIGN.md
```

---

## Tool Graph

### Required Node Types (5 total)

| Node Type | Description |
|---|---|
| `Tool` | Top-level API tool from ToolBench |
| `Endpoint` | A callable action within a tool |
| `Parameter` | An input parameter of an endpoint |
| `ResponseField` | An output field from an endpoint (when available) |
| `Concept` | Semantic domain tag (e.g. "travel", "weather", "finance") |

### Edge Types

| Edge | From → To | Meaning |
|---|---|---|
| `HAS_ENDPOINT` | Tool → Endpoint | Tool exposes this endpoint |
| `HAS_PARAMETER` | Endpoint → Parameter | Endpoint requires this input |
| `RETURNS` | Endpoint → ResponseField | Endpoint produces this output |
| `TAGGED_WITH` | Tool → Concept | Tool belongs to this domain |
| `COMPATIBLE_WITH` | Endpoint → Endpoint | Output of one feeds input of another |

`COMPATIBLE_WITH` is the key edge that enables realistic tool chaining.

### Sampler Pattern Types

| Pattern | Description | Build order |
|---|---|---|
| `multi_step` | Sequential — each step depends on prior output | Build first |
| `parallel` | Independent calls, results merged | Add in Phase 2 |
| `hybrid` | Mix of sequential and parallel | Add in Phase 2 |

### tacs build Artifacts

```
artifacts/
├── registry.pkl      # serialized ToolRegistry
├── graph.pkl         # serialized NetworkX DiGraph
└── build_meta.json   # {timestamp, tool_count, endpoint_count, node_count}
```

`tacs generate` loads these — it never re-parses ToolBench raw data.

---

## Core Data Models

### tacs/registry/models.py

```python
class Parameter(BaseModel):
    name: str
    type: str
    required: bool
    description: str = ""
    enum_values: list[str] = []
    default: Any = None

class Endpoint(BaseModel):
    name: str
    description: str
    method: str = "GET"
    parameters: list[Parameter] = []
    response_fields: list[str] = []

class Tool(BaseModel):
    tool_id: str
    name: str
    description: str
    category: str = "general"
    endpoints: list[Endpoint] = []
    source_data: dict = {}  # raw ToolBench data preserved for debugging
```

### tacs/agents/models.py

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str

class ToolCall(BaseModel):
    endpoint: str
    arguments: dict[str, Any]
    step: int

class ToolOutput(BaseModel):
    endpoint: str
    output: dict[str, Any]
    step: int

class ConversationMetadata(BaseModel):
    seed: int
    tool_ids_used: list[str]
    num_turns: int
    num_clarification_questions: int
    memory_grounding_rate: float | None  # null if only one tool call
    corpus_memory_enabled: bool

class Conversation(BaseModel):
    conversation_id: str
    messages: list[Message]
    tool_calls: list[ToolCall]
    tool_outputs: list[ToolOutput]
    metadata: ConversationMetadata
```

---

## MemoryStore Interface

**Do not change these method signatures — specified exactly by the assessment.**

```python
class MemoryStore:
    def add(self, content: str, scope: str, metadata: dict) -> None:
        """Store an entry. scope is 'session' or 'corpus'."""
        ...

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        """Return top_k entries from this scope only. Never cross scopes."""
        ...
```

### mem0 Initialization — inside MemoryStore ONLY

```python
from mem0 import Memory
m = Memory()  # embedded Qdrant — no external service needed
```

Never import mem0 anywhere outside `tacs/memory/store.py`.

### Scope Isolation Strategy

mem0 has no native scopes. Use metadata filtering:
- Every `add()` stores `{"scope": scope_value, ...}` in metadata
- Every `search()` filters to only return entries where `metadata["scope"] == scope`

### memory_grounding_rate — Exact Definition

```
memory_grounding_rate = (
    non-first-step tool calls with ≥1 memory entry retrieved
) / (
    total non-first-step tool calls
)
```

- `1.0` → every eligible call was grounded in memory
- `0.0` → none were grounded
- `null` → only one tool call in the conversation (nothing eligible)
- Count retrieval as present when `search()` returns at least one result (no score threshold)

### Memory Prompt Templates — Exact Format

**Session memory** (before argument-filling for any non-first tool call):
```
[Memory context]
{retrieved_entries}

Given the above context and the current tool schema, fill in the arguments for
{endpoint_name}.
```

**Corpus memory** (before Planner generates a new plan):
```
[Prior conversations in corpus]
{retrieved_summaries}

Given the above, plan a new diverse conversation using the following tool chain:
{proposed_tool_chain}
```

### Write Paths

**Session** — after every tool call completes:
```python
memory.add(
    content=json.dumps(tool_output),
    scope="session",
    metadata={"conversation_id": ..., "step": ..., "endpoint": ...}
)
```

**Corpus** — after each conversation is fully generated and validated:
```python
# e.g. "Tools: weather_api, maps_api. Domain: travel. Pattern: sequential multi-step."
memory.add(
    content=summary_text,
    scope="corpus",
    metadata={"conversation_id": ..., "tools": [...], "pattern_type": ...}
)
```

---

## MockExecutor — Determinism Strategy

Tool outputs must be mocked but deterministic and chain-consistent:

- Hash `seed + endpoint_name + json.dumps(arguments, sort_keys=True)` to seed fake value generation
- Same inputs → same outputs, always, across runs
- IDs from step 1 (e.g. `flight_id: "F99"`) stored in `SessionState`
- Later steps read `SessionState` to reference those IDs — never invent new ones

---

## Multi-Agent Pipeline Flow

```
For each conversation:

1. SamplerAgent     → proposes tool chain from ToolGraph (MUST use graph — never hardcoded)
2. PlannerAgent     → reads corpus memory → creates scenario + context
3. [Conversation loop]
   UserProxyAgent   → generates next user utterance
   AssistantAgent   → decides: clarify OR call tool
     if clarify:
       num_clarification_questions += 1
       loop back to UserProxyAgent for answer
     if tool call:
       MockExecutor   → validates args against schema, returns deterministic mock response
       SessionMemory  → write: store tool output
       (next tool call) SessionMemory → read: inject into argument-filling prompt
4. ValidatorAgent   → validates full conversation meets hard requirements
5. CorpusMemory     → write: store conversation summary
6. Serialise Conversation model → append to JSONL
```

### Conversation Hard Requirements (checked by ValidatorAgent + metrics)

- **Multi-step**: ≥ 3 tool calls in a substantial portion of dataset
- **Multi-tool**: ≥ 2 distinct tools in a substantial portion of dataset
- **Disambiguation**: AssistantAgent MUST ask clarifying questions when intent is
  ambiguous or required parameters are missing
- **Graph-first**: tool chains MUST come from the graph sampler — never a hardcoded list

---

## CLI Commands

```bash
# 1. Build registry, graph, index from ToolBench data
tacs build --data-dir data/ --output-dir artifacts/

# 2. Generate conversations (corpus memory ON by default = Run B)
tacs generate --seed 42 --count 50 --output output/run_b.jsonl

# 2b. Run A: corpus memory OFF (diversity experiment)
tacs generate --seed 42 --count 50 --no-corpus-memory --output output/run_a.jsonl

# 3. Validate a dataset
tacs validate --input output/run_b.jsonl

# 4. Metrics on one run
tacs metrics --input output/run_b.jsonl

# 4b. Diversity experiment comparison
tacs metrics --input-a output/run_a.jsonl --input-b output/run_b.jsonl
```

---

## Diversity Experiment

| Run | Corpus Memory | Flag |
|-----|--------------|------|
| A | Disabled | `--no-corpus-memory` |
| B | Enabled | (default) |

Same seed for both. Compute at least one metric:
- Pairwise tool-chain Jaccard dissimilarity ← recommended, easiest to implement
- n-gram distinct-N on assistant utterances
- Entropy over domain/pattern-type labels

Results go in `tacs metrics` CLI output AND in `DESIGN.md` → "Corpus Memory & Diversity Analysis".

---

## Output Format (JSONL)

One JSON object per line:

```json
{
  "conversation_id": "conv_00042",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "tool_calls": [
    {"endpoint": "flight_search", "arguments": {"origin": "NYC", "destination": "Tokyo"}, "step": 1}
  ],
  "tool_outputs": [
    {"endpoint": "flight_search", "output": {"flight_id": "F99", "price": 870}, "step": 1}
  ],
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["flight_api", "booking_api"],
    "num_turns": 6,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 1.0,
    "corpus_memory_enabled": true
  }
}
```

---

## LLM Client (tacs/llm.py)

```python
class LLMClient:
    def complete(self, messages: list[dict], **kwargs) -> str: ...
```

Backends (set via `TACS_LLM_BACKEND` env var): `ollama` (default), `openai`, `anthropic`.

**Default is ollama — pipeline must run without any paid API key.**

---

## Testing

```bash
pytest tests/ -v
```

| File | Tests |
|---|---|
| `tests/unit/test_registry.py` | Loading, normalization, missing field handling |
| `tests/unit/test_memory.py` | `add` → `search` returns entry; scope isolation |
| `tests/unit/test_graph.py` | Graph builds; sampler returns valid chain |
| `tests/e2e/test_pipeline.py` | Full build + generate 50 samples end-to-end |

---

## DESIGN.md Required Sections

```
## Architecture Overview
## Tool Registry Design
## Tool Graph Design            ← node types, edge types, sampler patterns
## Offline Execution Model      ← determinism strategy, session state
## Multi-Agent System Design    ← each agent's role and prompt strategy
## Memory System Design         ← scope isolation, grounding rate calculation
## Corpus Memory & Diversity Analysis   ← REQUIRED: metric chosen, Run A result,
                                           Run B result, 3-5 sentence analysis
## Design Decisions + Trade-offs
```

---

## Hard Rules — Never Break These

1. **Never use `print()`** — use `logger = logging.getLogger(__name__)`
2. **Never hardcode values** — all config lives in `tacs/config.py`
3. **Never import mem0 outside `tacs/memory/store.py`**
4. **Never pass raw dicts between modules** — use Pydantic models at boundaries
5. **Never use `random` without the seeded rng** — `rng = random.Random(config.seed)`
6. **Never catch bare `Exception`** without logging it with context
7. **Never start Phase 2 before Phase 1 runs end-to-end**

---

## Environment Variables (.env.example)

```bash
OPENAI_API_KEY=           # optional
ANTHROPIC_API_KEY=        # optional
OLLAMA_BASE_URL=http://localhost:11434
TACS_LLM_BACKEND=ollama   # default — no key needed
TACS_LOG_LEVEL=INFO
TACS_SEED=42
```

---

## Code Conventions

- Python 3.11+
- Type hints on every function signature
- Docstring on every class and public method
- Line length: 88 characters (black)
- Import order: stdlib → third-party → local (isort)
- `snake_case` files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
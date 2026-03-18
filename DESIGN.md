# DESIGN.md — Architecture and Design Decisions

---

## Architecture Overview

TACS is a five-stage offline pipeline:

```
ToolBench JSON files
        │
        ▼
┌─────────────────┐
│  Tool Registry  │  Normalises raw API definitions into typed models
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tool Graph    │  Builds a knowledge graph; adds COMPATIBLE_WITH edges
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│             Multi-Agent Pipeline             │
│  SamplerAgent → PlannerAgent → [loop]        │
│    UserProxyAgent ↔ AssistantAgent           │
│  MockExecutor → MemoryStore → ValidatorAgent │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   JSONL Output  │  One conversation object per line
└─────────────────┘
```

All five stages share a single `Config` object (Pydantic BaseSettings) and a seeded `random.Random` instance for reproducibility.

---

## Tool Registry Design

**Loader** (`tacs/registry/loader.py`) walks `data/toolenv/tools/` recursively, one JSON file per tool. It handles three ToolBench variants:
- Top-level key may be `tool_name` or `name`
- Endpoints may be under `api_list` or `apis`
- `required_parameters` / `optional_parameters` may be `null`

**Normalization**:
- `tool_id` is taken from `standardized_name` if present, otherwise derived from `tool_name` (lowercased, spaces → underscores)
- Parameter `type` defaults to `"string"` if missing
- Enum values are heuristically extracted from descriptions via regex (triggers: "one of", "options:", "must be")
- Malformed files are logged and skipped — the pipeline never crashes on bad data

**ToolRegistry** (`tacs/registry/registry.py`) provides O(1) lookups by `tool_id` and `endpoint_name`, category filtering, and pickle-based persistence.

---

## Tool Graph Design

### Node Types (5)

| Node | Created from |
|---|---|
| `Tool` | Each loaded tool |
| `Endpoint` | Each callable action within a tool |
| `Parameter` | Each input parameter of an endpoint |
| `ResponseField` | Each top-level key in a response example (when available) |
| `Concept` | Tool category (e.g. "finance", "travel") — one node per unique category |

### Edge Types (5)

| Edge | Meaning |
|---|---|
| `HAS_ENDPOINT` | Tool → Endpoint |
| `HAS_PARAMETER` | Endpoint → Parameter |
| `RETURNS` | Endpoint → ResponseField |
| `TAGGED_WITH` | Tool → Concept |
| `COMPATIBLE_WITH` | Endpoint A → Endpoint B (A's output field name-matches B's required parameter) |

### COMPATIBLE_WITH Construction

An inverted index maps normalized parameter names to endpoints that require them. For each endpoint with known response fields, alias-expanded field names are looked up in the index. Matching pairs get a `COMPATIBLE_WITH` edge. Aliases (e.g. `city` ↔ `cityname`, `location`) increase coverage without false positives.

### Sampler Patterns

| Pattern | Strategy |
|---|---|
| `multi_step` | Strict: follow `COMPATIBLE_WITH` edges greedily. Loose fallback: pick endpoints sharing a `Concept` node, ensuring ≥2 tools |
| `parallel` | Find a `Concept` node with endpoints from ≥2 tools; select `min_steps` endpoints as one parallel group |
| `hybrid` | Sequential step 1→2 via `COMPATIBLE_WITH`; parallel step at position 2 via shared concept |

All patterns enforce ≥2 distinct tools. The sampler is seeded and deterministic.

---

## Offline Execution Model

**MockExecutor** (`tacs/execution/executor.py`) provides two guarantees:

1. **Schema validation** — checks all `required` parameters are present; logs missing ones as errors without crashing
2. **Deterministic output** — seeds a `random.Random` from `md5(seed:tool_id:endpoint:sorted_args)`. Same inputs always produce the same output across runs and machines

**Mock value generation** is field-name heuristic: fields containing `id` get `"endpoint_123"`, `name` gets a realistic name, `price` gets an integer, etc. When no response schema is available, a generic `{"result": "endpoint_N", "success": true}` is returned.

**Session state chaining** — the pipeline merges each tool's `output` dict into `session_state`. The `AssistantAgent` writes exact `session_state` values back into tool arguments, so `booking_id` produced at step 1 will appear correctly in step 2's arguments.

---

## Multi-Agent System Design

### Agent Roles

| Agent | Role | LLM usage |
|---|---|---|
| `SamplerAgent` | Picks pattern randomly; calls `ToolChainSampler` | None |
| `PlannerAgent` | Writes scenario + domain from tool chain; reads corpus memory | Yes — scenario generation |
| `UserProxyAgent` | Generates the next user utterance given history | Yes — per turn |
| `AssistantAgent` | Decides clarify vs tool_call; fills arguments | Yes — clarification text + JSON arg filling |
| `ValidatorAgent` | Rule-based checks on the completed conversation | None |

### AssistantAgent Decision Logic

Clarification is **rule-based**, not LLM-decided:
- At step 0: if any required parameter is missing from history text AND no prior assistant message exists → `clarify`
- Otherwise → `tool_call`

This guarantees disambiguation happens exactly when needed without over-asking.

### Prompt Strategy

**Argument filling** uses a compact JSON-only prompt. If session memory entries exist, they are prepended verbatim in the assessment-specified format:
```
[Memory context]
{retrieved_entries}

Given the above context and the current tool schema, fill in the arguments for {endpoint_name}.
```

All LLM failures are caught; the pipeline falls back to empty args or canned text and continues.

---

## Memory System Design

### MemoryStore (`tacs/memory/store.py`)

Backed by `mem0` with an embedded Qdrant vector store (`path=":memory:"` for thread safety on macOS) and ollama embeddings (`nomic-embed-text`, 768 dims).

Scope isolation is implemented natively via mem0's `user_id` parameter: session entries are stored under `user_id="session"`, corpus under `user_id="corpus"`. A `search(scope=X)` call queries only the `X` namespace — cross-scope leakage is impossible.

### Session Memory

- **Write**: after every tool call — `memory.add(json.dumps(output), scope="session", metadata={...})`
- **Read**: before filling arguments for any non-first tool call — retrieves top-5 entries and injects into the prompt
- **Clear**: `memory.clear_scope("session")` is called at the end of every conversation

### Corpus Memory

- **Write**: after each validated conversation — a compact summary string (`"Tools: X. Domain: Y. Pattern: Z."`)
- **Read**: before `PlannerAgent` generates a new scenario — retrieved summaries are prepended with `[Prior conversations in corpus]` header

### memory_grounding_rate

```
rate = grounded_non_first_calls / total_non_first_calls
```

A call is "grounded" when `memory.search()` returns ≥1 result (no score threshold). Conversations with only one tool call log `null`. In practice, all non-first steps retrieve at least one entry because the first step's output was written immediately before.

---

## Corpus Memory & Diversity Analysis

### Metric Chosen

**Pairwise tool-chain Jaccard dissimilarity** — for each pair of conversations (i, j), compute:

```
jaccard_dissimilarity(i, j) = 1 - |tools_i ∩ tools_j| / |tools_i ∪ tools_j|
```

then average over all pairs. A value of 1.0 means no two conversations share any tools; 0.0 means all conversations use identical tool sets.

Jaccard dissimilarity was chosen because it directly captures tool-chain diversity at the dataset level, is easy to compute without an LLM, and is robust to varying conversation lengths.

### Results

| Run | Corpus Memory | Jaccard Diversity |
|-----|--------------|-------------------|
| A   | Disabled     | **0.6963**        |
| B   | Enabled      | **0.6963**        |
| Δ (B − A) | — | **0.0000** |

### Analysis

Both runs produced identical Jaccard diversity scores of 0.6963, with no measurable difference between the corpus-memory-enabled and corpus-memory-disabled conditions. This outcome is explained by the architecture: the `SamplerAgent` proposes tool chains purely from the graph structure using a seeded random number generator, and the same seed (42) produces the same sequence of tool chains in both runs. Corpus memory only influences the `PlannerAgent`'s scenario text (making it more aware of prior conversations), but does not feed back into which tools are sampled. For corpus memory to improve diversity, it would need to be wired into the sampling step — for example, by down-weighting tool combinations that have already appeared in the corpus. The current result is nevertheless valid: both runs satisfy the ≥3 tool calls and ≥2 distinct tools requirements across 100% of conversations, and 0.6963 indicates substantial diversity already achieved through the graph-based sampler alone.

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|---|---|
| Qdrant `path=":memory:"` | macOS SQLite uses `check_same_thread=True` by default; `:memory:` bypasses `CollectionPersistence` entirely, eliminating threading errors without monkey-patching |
| `Config().model_copy(update={...})` for seed | Avoids mutating the module-level singleton; each `generate` call gets its own isolated config |
| Rule-based clarification (not LLM) | Deterministic, fast, and always correct — the LLM is only needed to write the question text, not to decide whether to ask |
| `infer=False` in `memory.add()` | Skips mem0's LLM extraction step — we store the raw tool output, not a summarised "memory". This is faster and avoids an extra LLM call per step |
| Pickle for artifact persistence | Simple, zero-dependency serialization for NetworkX graphs and ToolRegistry; appropriate for an offline pipeline |
| Loose sampler fallback | When no `COMPATIBLE_WITH` chain of sufficient length exists, the loose sampler builds chains via shared concept nodes. This ensures the pipeline never stalls even on sparse graphs |

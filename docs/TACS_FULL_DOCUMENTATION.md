# Tool-Augmented Conversation Simulator (TACS)
## Complete Technical Documentation

Version: 0.1.0
Repository: tool-augmented-conversation-simulator
Language: Python 3.11+

---

## 1. Executive Summary

TACS is an offline synthetic data generation system that creates multi-turn, multi-tool conversational traces for training and evaluating tool-using AI assistants.

It ingests ToolBench API definitions, constructs a typed tool registry, derives a compatibility graph between endpoints, samples realistic tool-call chains, simulates multi-agent conversations around those chains, executes deterministic mock tool calls, and writes JSONL datasets with metadata suitable for quality checks and comparative experiments.

Core outcomes:
- Reproducible synthetic conversation generation using seeded randomness.
- Structured tool usage traces with tool calls and tool outputs.
- Session and corpus memory integration through mem0.
- Rule-based validation against hard quality constraints.
- CLI-based workflow for build, generate, validate, and metrics.

---

## 2. Repository Layout

Top-level files and packages:

- `README.md`: User-level setup and run instructions.
- `DESIGN.md`: Architecture and design rationale.
- `pyproject.toml`: Packaging metadata, dependencies, CLI entrypoint.
- `requirements.txt`: Fully pinned dependency versions.
- `.env.example`: Environment variable template.
- `tacs/`: Main application package.
- `tests/`: Unit and end-to-end test suites.

Package structure:

- `tacs/config.py`: Environment-backed configuration model.
- `tacs/llm.py`: Backend-agnostic LLM client wrapper.
- `tacs/cli.py`: Click commands for end-to-end operations.
- `tacs/registry/`: ToolBench ingestion and normalized registry.
- `tacs/graph/`: Graph modeling, graph construction, tool-chain sampling.
- `tacs/agents/`: Multi-agent orchestration and conversation logic.
- `tacs/execution/`: Deterministic mock tool execution.
- `tacs/memory/`: mem0-backed memory store wrapper.

Empty namespace/init markers:
- `tacs/__init__.py`
- `tacs/agents/__init__.py`
- `tacs/graph/__init__.py`
- `tacs/memory/__init__.py`
- `tacs/execution/__init__.py`
- `tacs/registry/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/e2e/__init__.py`

---

## 3. End-to-End Dataflow

### 3.1 Build Phase

1. Load ToolBench JSON files from `data/toolenv/tools/<category>/*.json`.
2. Normalize raw records into strongly typed `Tool`, `Endpoint`, and `Parameter` models.
3. Persist registry artifact as `artifacts/registry.pkl`.
4. Build directed graph of tools, endpoints, parameters, response fields, and concepts.
5. Add `COMPATIBLE_WITH` edges by matching response-field names to required-parameter names with alias expansion.
6. Persist graph as `artifacts/graph.pkl`.
7. Write `artifacts/build_meta.json` with counts and timestamp.

### 3.2 Generation Phase

Per conversation:

1. Sample a tool chain pattern (`multi_step`, `parallel`, `hybrid`) from the graph.
2. Plan scenario/domain/pattern text from sampled chain (optionally with corpus memory context).
3. Run turn loop:
   - User proxy emits user message.
   - Assistant chooses clarify vs tool-call vs final response.
   - If tool-call: execute mock tool, write output to session memory, append tool output message.
4. Compute memory-grounding rate for non-first tool calls.
5. Validate conversation against hard constraints.
6. If corpus memory enabled, store compact conversation summary in corpus scope.
7. Clear session memory.
8. Serialize conversation as one JSON line.

### 3.3 Validation and Metrics Phase

- Validate command checks generated conversations line-by-line using the same validator logic.
- Metrics command computes descriptive quality metrics and tool-chain diversity (pairwise Jaccard dissimilarity).

---

## 4. Configuration System

File: `tacs/config.py`

`Config` inherits `pydantic_settings.BaseSettings` with:
- `env_file=".env"`
- `env_prefix="TACS_"`
- `extra="ignore"`

### 4.1 LLM/Backend Parameters

- `llm_backend` (default `ollama`)
- `llm_max_tokens` (default `4096`)
- `ollama_base_url` (default `http://localhost:11434`)
- `ollama_model` (default `llama3`)
- `ollama_embed_model` (default `nomic-embed-text`)
- `ollama_embed_dims` (default `768`)
- `openai_api_key` (default empty)
- `openai_model` (default `gpt-4o-mini`)
- `anthropic_api_key` (default empty)
- `anthropic_model` (default `claude-haiku-4-5-20251001`)

### 4.2 Generation Parameters

- `seed` (42)
- `conversation_count` (50)
- `min_tool_calls` (3)
- `min_distinct_tools` (2)
- `max_turns` (10)
- `max_retries` (50)

### 4.3 Paths and Operational Settings

- `data_dir` (`data`)
- `artifacts_dir` (`artifacts`)
- `output_dir` (`output`)
- `log_level` (`INFO`)
- `corpus_memory_enabled` (`True`)
- `memory_top_k` (`5`)

Global singleton: `config = Config()`.

Important implementation detail:
- The `generate` CLI command uses `Config().model_copy(update={"seed": seed})` to avoid mutating the module-level singleton.

---

## 5. LLM Abstraction

File: `tacs/llm.py`

Class: `LLMClient`

Purpose:
- Provide one `complete(messages, **kwargs)` interface across multiple providers.

Supported backends:
- `ollama`: `ollama.chat(...)`
- `openai`: `OpenAI.chat.completions.create(...)`
- `anthropic`: `Anthropic.messages.create(...)` with special handling for system message extraction.

Behavioral notes:
- Uses global `config` model names and keys.
- No retries, no streaming, no token accounting, no template system.
- Returns assistant text as plain string.

Potential mismatch to consider:
- Anthropic format expects non-system messages in provider-native schema; code passes role/content dictionaries after filtering system role, which should work with Anthropic SDK message schema but should be revalidated on version changes.

---

## 6. CLI Commands

File: `tacs/cli.py`

CLI framework: Click group `cli` with logging setup.

### 6.1 `tacs build`

Inputs:
- `--data-dir`
- `--output-dir`

Actions:
- Load tools via `load_tools(...)`.
- Build `ToolRegistry` and save as pickle.
- Build graph via `ToolGraphBuilder` and save as pickle.
- Write `build_meta.json` with:
  - `timestamp`
  - `tool_count`
  - `endpoint_count`
  - `node_count`
  - `edge_count`

Failure handling:
- Raises `ClickException` if no tools found.

### 6.2 `tacs generate`

Inputs:
- `--seed`
- `--count`
- `--output`
- `--artifacts-dir`
- `--corpus-memory/--no-corpus-memory`

Actions:
- Load registry and graph artifacts.
- Instantiate `LLMClient`, `MemoryStore`.
- Run pipeline generate function.

### 6.3 `tacs validate`

Inputs:
- `--input` JSONL path

Actions:
- Parse each line into `Conversation` model.
- Validate using `ValidatorAgent`.
- Print counts of valid vs invalid.
- Exit status 1 when any invalid conversations exist.

### 6.4 `tacs metrics`

Modes:
- Single input (`--input`)
- Comparative inputs (`--input-a` and `--input-b`)

Computed metrics:
- Average turns
- Average tool calls
- Average clarification questions
- Average memory grounding (excluding `None`)
- Share of multi-step conversations (`>=3` calls)
- Share of multi-tool conversations (`>=2` tools)
- Jaccard diversity of tool sets
- Delta for Run B minus Run A in comparative mode

Observation:
- README and code are currently aligned on existence of `validate` and `metrics` commands.

---

## 7. Registry Subsystem

### 7.1 Data Models

File: `tacs/registry/models.py`

- `Parameter`
  - Fields: `name`, `type`, `required`, `description`, `enum_values`, `default`
- `Endpoint`
  - Fields: `name`, `description`, `method`, `parameters`, `response_fields`
- `Tool`
  - Fields: `tool_id`, `name`, `description`, `category`, `endpoints`, `source_data`

All models use Pydantic with `extra="ignore"`.

### 7.2 Loader Logic

File: `tacs/registry/loader.py`

Primary function: `load_tools(data_dir: Path) -> list[Tool]`

Supported raw format variants:
- `tool_name` or `name`
- `tool_description` or `description`
- `api_list` or `apis`
- Handles `required_parameters` and `optional_parameters` being `None`

Normalization behavior:
- `tool_id` from `standardized_name` when available.
- Else derive as lowercase tool name with spaces replaced by underscore.
- Endpoint method uppercased.
- Parameter type defaults to `string` if missing.

Enum extraction:
- `_parse_enum_values(description)` checks for triggers: `one of`, `options:`, `must be`.
- Extracts quoted entries first, fallback to comma-separated phrase parse.

Response fields:
- `_load_response_fields(...)` reads optional examples from `data/toolenv/response_examples/<tool_id>/<endpoint>.json`.
- Stores top-level JSON keys as endpoint response fields.

Error resilience:
- Malformed JSON files are logged and skipped.
- Parsing failures are non-fatal; loader returns best-effort list.

### 7.3 Registry API

File: `tacs/registry/registry.py`

Class: `ToolRegistry`

Capabilities:
- `get_tool(tool_id)`
- `get_endpoint(tool_id, endpoint_name)`
- `list_tools()`
- `list_by_category(category)`
- `list_categories()`
- `all_endpoints()` returning `(tool_id, endpoint)` tuples

Stats:
- `tool_count`
- `endpoint_count`
- `category_count`

Persistence:
- `save(artifacts_dir)` -> `registry.pkl`
- `load(artifacts_dir)` -> `ToolRegistry`

Failure modes:
- Missing artifact file: `FileNotFoundError`
- Pickle read errors: logged and re-raised

---

## 8. Graph Subsystem

### 8.1 Graph Ontology

File: `tacs/graph/models.py`

Node types (`NodeType` enum):
- `TOOL`
- `ENDPOINT`
- `PARAMETER`
- `RESPONSE_FIELD`
- `CONCEPT`

Edge types (`EdgeType` enum):
- `HAS_ENDPOINT`
- `HAS_PARAMETER`
- `RETURNS`
- `TAGGED_WITH`
- `COMPATIBLE_WITH`

Node ID factory:
- `node_id(node_type, name, parent=None)` creates deterministic IDs.

### 8.2 Graph Builder

File: `tacs/graph/builder.py`

Class: `ToolGraphBuilder`

`build()` behavior:
1. Add Tool nodes.
2. Add category Concept nodes; connect via `TAGGED_WITH`.
3. Add Endpoint nodes; connect Tool->Endpoint via `HAS_ENDPOINT`.
4. Add Parameter nodes; connect Endpoint->Parameter via `HAS_PARAMETER`.
5. Add ResponseField nodes; connect Endpoint->ResponseField via `RETURNS`.
6. Compute compatibility edges between endpoints.

Compatibility algorithm:
- Normalize field and parameter names by lowercasing and removing separators.
- Expand aliases using `_ALIASES`, for example:
  - `city` <-> `cityname`, `location`, `place`
  - `id` <-> `identifier`, `key`
- Build inverted index of required parameter keys to endpoint IDs.
- For each endpoint response field key, connect source endpoint to endpoints requiring matching key.
- Skip self loops and duplicate edges per destination.

Persistence:
- `save(graph, artifacts_dir)` -> `graph.pkl`
- `load(artifacts_dir)` -> `networkx.DiGraph`

### 8.3 Tool Chain Sampler

File: `tacs/graph/sampler.py`

Model: `ToolChain`
- `steps`: `list[list[str]]` (supports parallel groups)
- `pattern`: `multi_step | parallel | hybrid`
- `tool_ids`: distinct ordered tools used
- `flat_steps` property flattens grouped steps left-to-right

Class: `ToolChainSampler`

Sampling methods:
- `sample(pattern, min_steps)` dispatches to specific strategy.

#### Multi-step

- First tries strict chaining via outgoing `COMPATIBLE_WITH` edges.
- If strict fails, tries loose concept-based chain from endpoints inside one concept.
- Requires at least `min_steps` endpoints and at least 2 distinct tools.

#### Parallel

- Select one concept.
- Choose endpoints from at least two tools within that concept.
- Construct one step containing multiple endpoints (parallel group).
- Ensures width >= `min_steps`.

#### Hybrid

- Step 1: endpoint (prefer one with compatibility neighbors).
- Step 2: sequential neighbor via `COMPATIBLE_WITH`.
- Also include concept-compatible endpoint in same second step for parallelism.
- Enforces at least 2 distinct tools.

Retries:
- Uses `config.max_retries` for bounded attempts.

Seeded determinism:
- Uses local `random.Random(seed)` for reproducible chain sampling.

---

## 9. Agent Subsystem

### 9.1 Base Agent

File: `tacs/agents/base.py`

Abstract class `BaseAgent` stores shared `LLMClient` and `Config` references and requires subclass `run()` implementation.

### 9.2 Agent Models

File: `tacs/agents/models.py`

Conversation primitives:
- `Message(role, content)`
- `ToolCall(endpoint, arguments, step)`
- `ToolOutput(endpoint, output, step)`

Conversation-level models:
- `ConversationMetadata`
  - `seed`
  - `tool_ids_used`
  - `num_turns`
  - `num_clarification_questions`
  - `memory_grounding_rate`
  - `corpus_memory_enabled`
- `Conversation`
  - `conversation_id`
  - `messages`
  - `tool_calls`
  - `tool_outputs`
  - `metadata`

Planning/action models:
- `ConversationPlan`
- `AssistantAction`
- `ValidationResult`

### 9.3 SamplerAgent

File: `tacs/agents/sampler_agent.py`

Responsibilities:
- Choose sampling pattern (randomized order unless explicitly provided).
- Invoke `ToolChainSampler` to produce graph-derived chains.
- Try fallback patterns if initial choice fails.

Guarantee:
- Never uses hardcoded tool lists.

### 9.4 PlannerAgent

File: `tacs/agents/planner_agent.py`

Responsibilities:
- Convert `ToolChain` into scenario plan.
- Optionally query corpus memory for prior conversation summaries.
- Prompt LLM to return JSON object with scenario/domain/pattern type.
- Parse JSON (handles fenced code blocks).
- If LLM fails, derive fallback plan by keyword-based domain mapping.

Prompt convention:
- Uses system prompt requiring JSON-only output.
- When corpus entries exist, prefixes prompt with `[Prior conversations in corpus]` block.

Fallback domain mapping examples:
- `flight` -> `travel`
- `stock` -> `finance`
- `music` -> `entertainment`

### 9.5 UserProxyAgent

File: `tacs/agents/user_proxy.py`

Responsibilities:
- Generate opening user message from plan scenario.
- On later turns, answer assistant clarifications naturally.

Behavior:
- First turn intentionally omits details to encourage clarifying question.
- If no assistant message exists on follow-up, emits generic fallback message.
- LLM failures return deterministic fallback text.

### 9.6 AssistantAgent

File: `tacs/agents/assistant_agent.py`

Responsibilities:
- Decide action at each step: `clarify`, `tool_call`, or `respond`.
- Resolve target endpoint from sampled chain node ID.
- Retrieve session memory context for non-first steps.
- Fill tool-call arguments via LLM-generated JSON.
- Merge exact values from session state into parsed arguments.

Decision policy (rule-based):
- At step 0, if required parameters appear missing and assistant has not clarified yet -> `clarify`.
- Otherwise -> `tool_call`.
- If all steps complete (`final=True`) -> closing assistant response.

Grounding logic:
- `grounded=True` when non-first-step memory search returns at least one entry.

Fallback logic:
- Endpoint resolution failure at step 0 -> clarify generic.
- Endpoint resolution failure later -> tool call with current session state as arguments.

### 9.7 ValidatorAgent

File: `tacs/agents/validator_agent.py`

Rule-based checks:
1. Minimum number of tool calls.
2. Minimum number of distinct tools (from metadata).
3. Conversation has at least one message.
4. Clarification presence tracked (`?` in assistant message) but not required error.
5. `memory_grounding_rate` within `[0, 1]` if not null.
6. Required metadata semantics (`num_turns > 0`, non-empty `tool_ids_used`).
7. Non-negative clarification count.
8. Sequential tool-call steps (`[0,1,2,...]`).

Output:
- `ValidationResult` with validity flag, errors, and summary stats.

### 9.8 Pipeline Orchestration

File: `tacs/agents/pipeline.py`

Class: `ConversationPipeline`

Execution sequence in `run(...)`:
1. Sample chain (`SamplerAgent`).
2. Create plan (`PlannerAgent`).
3. Conversation loop up to `max_turns`:
   - User message (`UserProxyAgent`).
   - Assistant action (`AssistantAgent`).
   - Optional tool execution (`MockExecutor`).
   - Session memory write and state update.
   - Record `ToolCall`, `ToolOutput`, and tool message.
4. Compute memory grounding rate:
   - `None` when no non-first tool calls.
   - Else grounded/non-first ratio.
5. Build `Conversation` object.
6. Validate with `ValidatorAgent`.
7. Write corpus summary memory when enabled and chain/plan available.
8. Clear session memory.

Error strategy:
- Catches unexpected pipeline exceptions, logs stack trace, clears session memory, and still returns a `Conversation` object built from collected state.

Batch generation function: `generate(...)`
- Creates output directory.
- Iterates `count` times with `tqdm` progress bar.
- Writes each conversation line immediately (stream-safe behavior).

---

## 10. Execution Subsystem

File: `tacs/execution/executor.py`

Models:
- `MockResult(output, valid, errors)`

Class: `MockExecutor`

`execute(tool_id, endpoint_name, arguments)`:
1. Validate endpoint exists in registry.
2. Validate required arguments are present.
3. Seed deterministic RNG from MD5 hash of:
   - global seed
   - tool ID
   - endpoint name
   - sorted JSON args
4. Generate response fields:
   - If schema fields exist: field-name-based heuristics.
   - Else fallback generic response.
5. Echo `_input` arguments into output for chain consistency.

Heuristic value generation examples:
- Field contains `id` -> `endpoint_123`
- Field contains `name` -> realistic person name
- `price`/`cost` -> integer
- `date` -> date string
- `url` -> fake URL
- `status` -> one of `active|pending|confirmed`

Determinism guarantee:
- Same inputs produce same outputs across runs.

---

## 11. Memory Subsystem

File: `tacs/memory/store.py`

Class: `MemoryStore`

Backend:
- `mem0` with Qdrant vector store configured as in-memory (`path=":memory:"`).

Scope model:
- `session` scope: per-conversation transient tool outputs.
- `corpus` scope: cross-conversation summaries.

Isolation mechanism:
- Uses `user_id=scope` directly in mem0 add/search/delete operations.

Backend-specific configuration:
- `ollama`: Ollama LLM + Ollama embedder.
- `openai`: OpenAI LLM + OpenAI embeddings (`text-embedding-3-small`, 1536 dims).
- `anthropic`: Anthropic LLM + OpenAI embeddings (Anthropic has no embeddings API).

Anthropic guardrail:
- Raises `ValueError` at initialization if backend is anthropic and `OPENAI_API_KEY` is absent.

API methods:
- `add(content, scope, metadata)`
  - Stores raw content with `infer=False`.
  - Swallows exceptions after logging.
- `search(query, scope, top_k)`
  - Returns list of dict entries from mem0 `results`.
  - Returns empty list on error.
- `clear_scope(scope)`
  - Deletes all entries for namespace.
  - Used by pipeline to reset session memory.

Operational implications:
- Memory operations are non-fatal; pipeline continuity preferred over strict failure.

---

## 12. Data Contracts and JSON Output

Each generated JSONL line follows `Conversation` schema.

Top-level fields:
- `conversation_id`: unique per run (`conv_00000`, ...)
- `messages`: ordered sequence of user/assistant/tool messages
- `tool_calls`: structured list with endpoint, args, and step
- `tool_outputs`: structured list with endpoint output and step
- `metadata`: generation and quality metadata

Metadata semantics:
- `tool_ids_used`: ordered distinct tools from sampled chain
- `num_turns`: count of all message objects in `messages`
- `memory_grounding_rate`:
  - `null` when only first-step call(s) exist
  - else ratio in `[0,1]`

---

## 13. Testing Strategy and Coverage

### 13.1 Unit Tests: Registry

File: `tests/unit/test_registry.py`

Coverage:
- Enum extraction parser behavior.
- Parameter parsing required/optional/default type.
- Endpoint parsing method normalization and response fields.
- Loader behavior on valid, malformed, and missing directories.
- Registry API lookups and aggregate counts.
- Registry pickle save/load lifecycle.

### 13.2 Unit Tests: Graph and Sampler

File: `tests/unit/test_graph.py`

Coverage:
- Graph type and node/edge type presence.
- `COMPATIBLE_WITH` existence in synthetic fixture.
- ToolChain flatten behavior for sequential/parallel/hybrid structures.
- Sampler outputs for all patterns.
- Constraints (`min_steps`, min 2 tools).
- Error conditions for invalid pattern and invalid min_steps.
- Determinism with same seed.

### 13.3 Unit Tests: Memory

File: `tests/unit/test_memory.py`

Coverage:
- add-search round trip returns results.
- Result shape checks.
- `top_k` limiting.
- Scope isolation between `session` and `corpus`.
- Scope clear behavior and non-interference.

Conditional skip behavior:
- Module skipped if `MemoryStore` cannot initialize (for example missing backend service or API key).

### 13.4 End-to-End Tests

File: `tests/e2e/test_pipeline.py`

Preconditions:
- `data/` must exist or tests are skipped.
- Expects artifacts and output files generated externally.

Coverage:
- Artifact existence and structural validity (`registry.pkl`, `graph.pkl`, `build_meta.json`).
- Graph minimum structural requirements.
- Dataset-level schema checks for Run A and Run B JSONL outputs.
- At least 50 conversations in each run.
- Corpus memory flag correctness per run.
- Basic quality checks (multi-step, multi-tool, tool calls/output count match, unique IDs).

Important note:
- E2E tests validate outputs and artifacts but do not invoke CLI commands internally.

---

## 14. Packaging and Dependencies

### 14.1 Build Metadata

File: `pyproject.toml`

- Project name: `tacs`
- Version: `0.1.0`
- Python requirement: `>=3.11`
- Console script: `tacs = tacs.cli:cli`

Tools configuration:
- Black line length `88`
- isort profile `black`
- pytest testpaths `tests`

### 14.2 Dependency Sets

High-level runtime dependencies include:
- Pydantic + pydantic-settings
- NetworkX
- Click
- mem0ai
- openai
- anthropic
- ollama
- python-dotenv
- faker
- tqdm

Pinned versions in `requirements.txt` provide exact reproducibility for environment recreation.

---

## 15. Operational Runbook

### 15.1 Initial Setup

1. Create and activate virtualenv.
2. Install dependencies from `requirements.txt`.
3. Install editable package (`pip install -e .`).
4. Configure `.env` from `.env.example`.
5. Ensure ToolBench data is available in expected layout.

### 15.2 Build Artifacts

Command:
- `tacs build --data-dir data --output-dir artifacts`

Expected artifacts:
- `artifacts/registry.pkl`
- `artifacts/graph.pkl`
- `artifacts/build_meta.json`

### 15.3 Generate Datasets

Run A (no corpus memory):
- `tacs generate --seed 42 --count 50 --no-corpus-memory --output output/run_a.jsonl`

Run B (corpus memory enabled):
- `tacs generate --seed 42 --count 50 --output output/run_b.jsonl`

### 15.4 Validate and Compare

- `tacs validate --input output/run_b.jsonl`
- `tacs metrics --input output/run_b.jsonl`
- `tacs metrics --input-a output/run_a.jsonl --input-b output/run_b.jsonl`

---

## 16. Determinism and Reproducibility

Deterministic components:
- Tool-chain sampler seeded with `Config.seed`.
- Mock executor outputs seeded from deterministic hash.

Non-deterministic influences:
- LLM responses (planner/user/assistant) can vary by backend model behavior.
- Memory retrieval ranking may vary with embedding/index implementation details.

Practical reproducibility expectations:
- Structural trace (tool-chain, step counts, deterministic mock outputs) is stable for fixed artifacts and seed.
- Natural language content can vary unless deterministic model settings are enforced externally.

---

## 17. Error Handling Philosophy

General principle:
- Prefer graceful degradation over hard failure for generation path.

Examples:
- Malformed tool JSON files are skipped during load.
- Memory add/search failures log warnings and continue.
- LLM parse failures fall back to default plans/messages/arguments.
- Pipeline catches unexpected exceptions, clears session memory, still returns conversation object.

Hard failures where appropriate:
- Missing build artifacts for commands that require them.
- Missing tools in build stage triggers ClickException.
- Validation command exits non-zero when invalid conversations found.

---

## 18. Known Constraints and Trade-offs

1. Pickle artifact format:
   - Fast and simple but language/runtime-coupled.
2. Compatibility matching by name heuristics:
   - Efficient but can miss semantic links not reflected in parameter names.
3. Clarification detection in validator:
   - Uses question-mark heuristic, may under/over count edge cases.
4. Memory fallback behavior:
   - Errors are suppressed to keep pipeline running; hidden degradation is possible without monitoring logs.
5. E2E tests are artifact-dependent:
   - They do not generate data themselves; CI needs pre-generated assets or adapted setup.
6. LLM prompt contracts are strict JSON text:
   - Robust parsing strips fenced blocks, but malformed output still falls back.

---

## 19. Extension Points

Recommended extension opportunities:

1. Replace mock execution with real tool adapters:
   - Add interface in execution layer for live API invocation and sandboxing.
2. Integrate memory-aware sampler:
   - Feed corpus memory into tool-chain sampling to directly optimize diversity.
3. Add retry and backoff in `LLMClient`:
   - Improve robustness for transient backend failures.
4. Add richer validation checks:
   - Verify semantic alignment between scenario and tool calls.
5. Persist non-pickle graph/registry exports:
   - JSON or parquet snapshots for easier inspection and interoperability.
6. Add pipeline-level integration tests:
   - Execute CLI end-to-end in temporary fixtures.

---

## 20. Security and Data Considerations

- Generated outputs are synthetic and mock execution does not call real APIs by default.
- API keys may be used for LLM and embedding backends; store in `.env` and avoid logging secrets.
- Pickled artifacts should be treated as trusted-only inputs due to pickle deserialization risks.

---

## 21. File-by-File Index

### Top-level
- `.env.example`: Environment template.
- `README.md`: User guide.
- `DESIGN.md`: Design narrative and experiment framing.
- `pyproject.toml`: Package + tooling config.
- `requirements.txt`: Pinned dependency lock-style list.

### Package: `tacs`
- `tacs/config.py`: Settings model and global config.
- `tacs/llm.py`: LLM backend abstraction.
- `tacs/cli.py`: Build/generate/validate/metrics commands.

### Package: `tacs.registry`
- `tacs/registry/models.py`: Tool/Endpoint/Parameter schemas.
- `tacs/registry/loader.py`: ToolBench parser and normalizer.
- `tacs/registry/registry.py`: Lookup index and persistence.

### Package: `tacs.graph`
- `tacs/graph/models.py`: Node/edge enums and node-id helper.
- `tacs/graph/builder.py`: Graph construction and compatibility edge creation.
- `tacs/graph/sampler.py`: Pattern-aware chain sampling.

### Package: `tacs.agents`
- `tacs/agents/base.py`: Abstract agent contract.
- `tacs/agents/models.py`: Conversation and action Pydantic models.
- `tacs/agents/sampler_agent.py`: Chain sampling agent wrapper.
- `tacs/agents/planner_agent.py`: Scenario planning with optional corpus memory.
- `tacs/agents/user_proxy.py`: User utterance simulator.
- `tacs/agents/assistant_agent.py`: Clarify/tool-call/respond decision and argument filling.
- `tacs/agents/validator_agent.py`: Rule-based conversation validator.
- `tacs/agents/pipeline.py`: Full orchestration and JSONL writer.

### Package: `tacs.execution`
- `tacs/execution/executor.py`: Deterministic mock tool execution.

### Package: `tacs.memory`
- `tacs/memory/store.py`: mem0 wrapper with strict scope isolation.

### Tests
- `tests/unit/test_registry.py`: Registry parser/loader/registry behavior.
- `tests/unit/test_graph.py`: Graph and sampler checks.
- `tests/unit/test_memory.py`: Memory add/search/isolation/clear.
- `tests/e2e/test_pipeline.py`: Artifact and dataset-level output validation.

---

## 22. Quick Reference Commands

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

Build:

```bash
tacs build --data-dir data --output-dir artifacts
```

Generate:

```bash
tacs generate --seed 42 --count 50 --output output/run_b.jsonl
tacs generate --seed 42 --count 50 --no-corpus-memory --output output/run_a.jsonl
```

Validate:

```bash
tacs validate --input output/run_b.jsonl
```

Metrics:

```bash
tacs metrics --input output/run_b.jsonl
tacs metrics --input-a output/run_a.jsonl --input-b output/run_b.jsonl
```

Test:

```bash
pytest tests -v
```

---

## 23. Conclusion

TACS is a modular, practical synthetic data generation framework with explicit abstractions for tool ingestion, graph-based tool planning, multi-agent conversation generation, deterministic execution, and memory-aware behavior.

Its architecture is intentionally pragmatic: graph and validation logic are deterministic and inspectable, while natural language generation is delegated to interchangeable LLM backends. The current implementation already supports reproducible offline experiments and quality checks, and it provides clear extension points for richer realism, stronger evaluation, and production-scale dataset generation.

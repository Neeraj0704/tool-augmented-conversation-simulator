# Multi-Agent Tool-Use Conversation Generator

An offline synthetic data generation system that produces multi-turn, multi-tool conversation traces for training and evaluating tool-use AI agents. Built on top of [ToolBench](https://github.com/OpenBMB/ToolBench) API definitions.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with:
  - `ollama pull llama3` — LLM for conversation generation
  - `ollama pull nomic-embed-text` — embeddings for memory
- ToolBench data (see Setup)

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd tool-augmented-conversation-simulator
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed — defaults work with a local Ollama instance
```

### 3. Download ToolBench data

Download the ToolBench dataset and place it at `data/toolenv/tools/` with the structure:

```
data/
└── toolenv/
    └── tools/
        ├── Finance/
        │   └── *.json
        ├── Entertainment/
        │   └── *.json
        └── ...
```

---

## Running End-to-End

### Step 1 — Build registry and graph

```bash
tacs build --data-dir data/ --output-dir artifacts/
```

Produces `artifacts/registry.pkl`, `artifacts/graph.pkl`, `artifacts/build_meta.json`.

### Step 2 — Generate conversations

**Run B** (corpus memory enabled — default):
```bash
tacs generate --seed 42 --count 50 --output output/run_b.jsonl
```

**Run A** (corpus memory disabled — for diversity experiment):
```bash
tacs generate --seed 42 --count 50 --no-corpus-memory --output output/run_a.jsonl
```

### Step 3 — Validate the dataset

```bash
tacs validate --input output/run_b.jsonl
```

### Step 4 — Compute metrics

Single run:
```bash
tacs metrics --input output/run_b.jsonl
```

Diversity experiment (Run A vs Run B):
```bash
tacs metrics --input-a output/run_a.jsonl --input-b output/run_b.jsonl
```

---

## Running Tests

```bash
source venv/bin/activate
pip install pytest
pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `tests/unit/test_registry.py` | ToolBench loading, normalization, missing field handling |
| `tests/unit/test_memory.py` | `add→search` round-trip, scope isolation (session vs corpus) |
| `tests/unit/test_graph.py` | Graph construction, all node/edge types, sampler patterns |
| `tests/e2e/test_pipeline.py` | Build artifacts + validate ≥50 generated conversations |

---

## Configuration

All settings are controlled via environment variables with the `TACS_` prefix (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `TACS_LLM_BACKEND` | `ollama` | LLM backend: `ollama`, `openai`, `anthropic` |
| `TACS_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TACS_OLLAMA_MODEL` | `llama3` | Chat model for generation |
| `TACS_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for memory |
| `TACS_SEED` | `42` | Random seed for reproducibility |
| `TACS_CONVERSATION_COUNT` | `50` | Number of conversations to generate |
| `TACS_MAX_TURNS` | `10` | Maximum turns per conversation |
| `OPENAI_API_KEY` | _(optional)_ | Required only if `TACS_LLM_BACKEND=openai` |
| `ANTHROPIC_API_KEY` | _(optional)_ | Required only if `TACS_LLM_BACKEND=anthropic` |

---

## Output Format

Each line in the JSONL output is a complete conversation:

```json
{
  "conversation_id": "conv_00042",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "tool_calls": [
    {"endpoint": "flight_search", "arguments": {"origin": "NYC"}, "step": 0}
  ],
  "tool_outputs": [
    {"endpoint": "flight_search", "output": {"flight_id": "F99"}, "step": 0}
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

## Project Structure

```
tacs/
├── config.py          # Central config (Pydantic BaseSettings)
├── llm.py             # LLMClient abstraction
├── cli.py             # Click CLI: build / generate / validate / metrics
├── registry/          # Tool Registry (ToolBench ingestion)
├── graph/             # Tool Graph + ToolChainSampler
├── execution/         # MockExecutor (deterministic mock responses)
├── agents/            # Multi-agent pipeline (5 agents)
└── memory/            # MemoryStore backed by mem0
```

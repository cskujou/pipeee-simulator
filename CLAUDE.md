## Project Overview

PipeEE is an LLM Engine implementing speculative decoding with early exit and pipeline parallelism. The codebase simulates how a speculative decoder with multiple draft models (early exit layers) would process requests.

**Note: This project must be managed with `uv`.** All dependency installation, script execution, and package management should use `uv` commands (e.g., `uv run`, `uv pip install`, `uv sync`). Do not use `pip` or other package managers.

## Architecture

### High-Level Flow

```
LLM (llm.py)
├── PrefillEngine (engine.py)     # Tokenizes prompts, runs first forward pass
└── DecodeEngine (engine.py)
    ├── Scheduler (sched/scheduler.py)    # Manages waiting/running queues, schedules batches
    └── Executor (executor.py)            # Executes model across pipeline-parallel workers
        ├── Worker (worker/worker.py)      # Processes subset of transformer layers
        │   └── ModelRunner (worker/model_runner.py)  # Forward pass for assigned layers
        └── Buffer (buffer.py)            # Inter-worker communication
```

### Core Concepts

**Speculative Decoding**: Instead of generating one token at a time, the system speculatively generates multiple candidate tokens (spec_topk) at each step, building a tree of possibilities. The draft tokens are verified in subsequent passes.

**Early Exit**: Intermediate layers can exit early with speculative tokens, creating multiple candidate paths. The final layer accepts/rejects these candidates.

**Tree Structure (EarlyExitedTokenTrie)**: Each Request contains a trie that stores:
- The root node (accepted tokens)
- Child nodes (speculative candidates with confidence scores)
- Position tracking for each node
- Cache IDs for KV cache management

**Pipeline Parallelism**: The model is split across `pp_size` workers. Each worker processes a subset of layers (e.g., 4 layers per worker for a 16-layer model). Workers communicate via buffers, passing intermediate hidden states.

**KV Cache System**:
- `kv_cache`: Standard transformers cache for completed tokens
- `spec_cache`: Separate cache for speculative token KV values, indexed by position

**Two-Stage Attention**: In `LlamaAttention`, attention uses:
1. Completed key/value states from `kv_cache`
2. Speculative key/value states from `spec_cache`
Both are concatenated before computing attention.

### Key Files

| File | Purpose |
|------|---------|
| `llm.py` | High-level LLM interface orchestrating prefill/decode |
| `engine.py` | PrefillEngine and DecodeEngine implementations |
| `sched/scheduler.py` | Request scheduling with waiting/running queues |
| `request_trie.py` | EarlyExitedTokenTrie - tree structure for speculative paths |
| `request.py` | Request class containing trie and generation state |
| `worker/models/modeling_llama.py` | LlamaForCausalLM with spec_cache support in attention |
| `sched/spec_cache.py` | SpecCache for storing speculative KV values |

### Scheduler Budget

The scheduler uses pluggable budget strategies (`SchedulerBudgetBase`):
- `DefaultSchedulerBudget`: Simple batch size limiting
- `BMCSchedulerBudget`: Memory and compute budget aware (partial implementation)

### Entry Points

```bash
uv run evaluate/run1.py
```

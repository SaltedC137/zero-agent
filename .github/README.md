# ZATO — Zero-Agent Tool Orchestrator

Local LLM agent in C++23 on [llama.cpp](https://github.com/ggml-org/llama.cpp). Tool calling, sandboxed bash, persistent memory, coloured terminal REPL.

## Quick Start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./out/ZATO --agent
```

## Usage

```
./out/ZATO --agent [--session <name>] [--system-prompt <path>] [model.gguf]
```

| Flag | Purpose |
|------|---------|
| `--agent` | Enable tool-calling agent mode |
| `--session <name>` | Named persistent session (default: `default`) |
| `--system-prompt <path>` | Override system prompt |
| `--list-sessions` | Show saved sessions |
| `exit` | Quit and save session |

## Tools

| Tool | Description |
|------|-------------|
| `add` | Integer addition |
| `read_text_file` | Read project files (relative paths, 4 KB default) |
| `echo` | Echo text |
| `run_bash` | Sandboxed shell commands |

### `run_bash` sandbox (via [bubblewrap](https://github.com/containers/bubblewrap))

System dirs read-only, project dir read-write, network disabled, PID isolation, clean env. 30 s timeout, 10 KB output limit. Commands require `[y/N]` confirmation. `rm`/`sudo`/`chmod`/`mkfs`/fork bombs are blocked. Falls back to bare subprocess when bwrap is missing.

## Features

**Session persistence** — conversations auto-saved to `.zato/sessions/<name>/` next to the binary as OpenAI-compatible JSON + KV cache. Restart with `--session` to resume.

**Context management** — old messages auto-trimmed when total tokens exceed 80% of `n_ctx` (16384), preserving system prompt and recent turns.

**Bash block interception** — when the 3B model outputs ` ```bash ` blocks instead of calling `run_bash`, the agent extracts and executes them automatically.

**ANSI colours** — green `You>`, magenta `AI>`, yellow command review, red errors.

**IModel interface** — Agent depends on an abstract `IModel`, decoupled from llama.cpp. Ready for API backends (OpenAI, Anthropic).

## Architecture

```
gym.cpp (REPL)
  ├── Agent            ← tool loop + callback hooks
  ├── IModel            ← abstract backend
  │     └── Model        ← llama.cpp GGUF
  ├── Tool              ← Echo, Add, ReadTextFile, RunBash
  ├── ToolRegistry      ← thread-safe factory
  ├── SessionManager    ← JSON + KV cache persistence
  └── ContextManager    ← token-aware window trimming
```

## Dependencies

- **llama.cpp** — local GGUF inference
- **cpp-httplib** — HTTP client (API backends)
- **nlohmann/json** — JSON
- **bubblewrap** (optional) — sandboxed execution
- C++23 (GCC 14+ / Clang 18+)

## Config

Defaults in `src/gym.cpp`: `temp=0.0`, `top_k=1`, `n_ctx=16384`, `n_batch=2048`. Set `n_gpu_layers=999` for full GPU offload. Pass `-DZATO_USE_CUDA_BACKEND=ON` to CMake for CUDA.

## License

Copyright (c) 2026 Aska Lyn

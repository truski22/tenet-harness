# tenet-harness

A small, model-agnostic runtime for building LLM agents that behave predictably: bounded runs,
explicit tool contracts, per-turn tool routing, durable event logs and a deterministic evaluation
toolkit. Python, built on LangGraph.

> **Status: 0.1, extracted from a working application.** [second-brain](https://github.com/truski22/second-brain)
> is the reference consumer. The public API may still change between minor versions. See the
> [roadmap](#roadmap) for what is built and what is not.

## What it does today

| Concern | What the harness provides |
| --- | --- |
| Tool contracts | `Tool`, `Toolset`, `Effect` (read / write / destructive), validated pydantic inputs, `ToolResult` with explicit status |
| Bounded runs | Tool-call budget, per-run deadline, per-run cooperative cancellation, no automatic replay of side-effecting tools |
| Concurrency | `max_concurrent_runs` (one active run per conversation) and a shared `ModelGate` that limits simultaneous model calls; `Runtime.exclusive()` for safe model switches |
| Context control | Per-turn toolset routing (keyword and embedding routers), history trimming to a token budget, `use_toolset` meta-tool for on-demand tools |
| Untrusted data | Toolsets marked `external` wrap results as `untrusted_data`; the registry rejects non-read tools on external toolsets |
| Durability | `RunStore` protocol (SQLite and in-memory implementations), replayable events by sequence number, LangGraph checkpoints, idempotent run IDs, interrupted-run recovery |
| Evaluation | Rule-based scoring (no LLM judge), pass@1 with Wilson intervals, pass^k reliability, critical-failure gate, baseline comparison |
| Testing | `tenet_harness.testing.FakeProvider` for deterministic, offline agent tests |
| Localisation | Runtime texts are injected through `Messages`; error codes are stable |

## Quickstart

```bash
uv sync --extra integrations
uv run python examples/echo_agent.py   # scripted model, no Ollama needed
uv run pytest -q
```

A complete agent is a few lines (see [`examples/echo_agent.py`](examples/echo_agent.py)):

```python
echo = Tool("echo", "Return the supplied text.", EchoInput,
            lambda context, args: ToolResult(status="success", data=args.text), Effect.READ)
agent = AgentDefinition(
    id="echo", name="Echo", description="Example agent.",
    toolsets=(Toolset("echo", "Echo text back.", (echo,), always_on=True),),
    context=lambda context: SystemMessage(content="Use only your registered tools."),
)
```

Toolset fields: `summary` (shown by `use_toolset`), `keywords` / `examples` (routing),
`instructions` (appended to the prompt only when active), `max_result_chars` (results are truncated),
`available` (unconfigured toolsets are never bound) and `external`. Raise `ToolError` for expected
failures; unexpected exceptions become a generic error result and never leak details to the model.

## Package layout

```
tenet_harness/
  tools.py  agents.py  runs.py  errors.py  messages.py   # contracts: Tool/Toolset, AgentDefinition/Registry, requests and events
  runtime/      engine (Runtime facade) · executor (one run) · binding (toolset selection) · tool_executor
                graph (agent loop) · gate (model concurrency) · events · history · memory (background worker)
  providers/    Provider protocol · Ollama adapter
  routing/      Router protocol · keyword, embedding and combined routers
  storage/      RunStore protocol · SQLite implementation
  evaluation/   scoring · stats · report
  integrations/ Integration protocol, secrets
  testing.py    FakeProvider, InMemoryStore
```

## Design decisions

- **Deterministic evaluation, no LLM judge.** Rules check observable behaviour (tools called,
  arguments, resulting state, grounding of numbers). Cheaper, reproducible, and it fails loudly.
- **Tool allow-lists are code-level boundaries, not a sandbox.** Do not load untrusted third-party tools.
- **Run events expose actions and status, not hidden reasoning.**
- **Concurrency is opt-in.** Defaults (1 run, 1 model call) match a single-user local app. Runs are
  threads inside one process; scaling out needs a shared store and checkpointer (see roadmap).

## Roadmap

Done and still open, in order:

1. ~~Split `Runtime._run`; per-run cancellation and concurrent runs; storage protocol~~ (done in 0.1)
2. Typed run events, `mypy --strict`, provider capabilities and token usage
3. Provider adapters beyond Ollama (Anthropic, OpenAI-compatible) and a cross-provider eval report
4. Policy-as-code and a human-approval checkpoint for write / destructive tools
5. Loop and no-progress detection; token and cost budgets
6. MCP client / server adapters
7. OpenTelemetry traces and metrics
8. Postgres store, container image, Helm chart

## License

MIT

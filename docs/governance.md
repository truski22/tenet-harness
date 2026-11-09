# Governance: policy, approval, loop detection, budgets, audit

Every tool call passes through the same pipeline in `ToolExecutor`:

```
model asks for a tool
  -> arguments validated (pydantic)            invalid -> clarification result, nothing runs
  -> loop guard                                 repeated call -> refused, model told to stop
  -> policy.evaluate(tool, toolset, args)       allow | deny | require_approval
        deny              -> error result for the model; audited
        require_approval  -> run pauses until a person decides (approve / reject / expire)
        allow             -> tool runs
  -> result bounded to the toolset's character budget, wrapped as untrusted_data if the toolset is external
  -> audit: tool_decision, [approval_*], tool_outcome
```

## Policy as code

A policy is data, validated when loaded, and can be reviewed and tested like code
(see [`examples/policy.yaml`](../examples/policy.yaml)).

```yaml
version: 1
default: allow
rules:
  - id: production-is-read-only
    tools: ["gitlab_*"]            # glob on tool names
    toolsets: ["*"]                # glob on toolset ids
    effects: [write, destructive]  # any effect if omitted
    when:                          # all must hold; a missing argument never matches
      - {arg: project, op: matches, value: "^prod/"}
    decision: deny
    reason: Production projects are read-only for agents.
```

- Rules are evaluated in order; **the first match wins**.
- Operators: `eq`, `ne`, `in`, `not_in`, `matches` (regular expression). `arg` accepts dotted paths.
- **Baseline.** If no rule matches, write or destructive tools of an `external` toolset require approval.
  Everything else follows `default`. An explicit `allow` rule overrides the baseline, so relaxing it is a
  visible diff in a reviewed file, never an accident.
- Unknown keys, duplicate rule ids, bad regular expressions and wrong value types are rejected at load time
  with `PolicyError`.

```python
runtime = Runtime(registry, provider, store, policy=Policy.from_yaml(Path("policy.yaml")))
```

## Human approval

When a call requires approval the run enters the `awaiting_approval` status and emits an
`approval_required` event. Any caller (CLI, web hook, chat bot) resolves it:

```python
for request in runtime.pending_approvals():
    runtime.resolve_approval(request["id"], approve=True, actor="alice", note="checked ticket")
```

- A request is resolved **exactly once**; a second attempt raises `RunRejected("APPROVAL_NOT_PENDING")`.
- Unanswered requests expire after `approval_timeout` (default 600 s). A rejected or expired action is
  reported to the model as an error with the instruction not to retry or work around it.
- Cancelling the run cancels the request. The run's deadline does not advance while it waits for a person.
- Arguments shown to the approver and stored in the audit log pass through `redact`
  (default: values of keys that look like credentials are masked). The tool itself receives the real values.
- **Not durable across a process restart.** Waiting is in-process. On restart, pending requests are marked
  `expired`, the run is marked `interrupted`, and the interrupted tool call is never replayed.

## Loop and failure detection

`LoopPolicy` (per run, `None` disables it):

| Setting | Default | Effect |
| --- | --- | --- |
| `max_identical_calls` | 3 | The same tool with the same arguments may execute this many times; the next attempt is refused and the model is told to answer with what it has |
| `max_blocked_calls` | 2 | After this many refusals the run stops with `loop_detected` |
| `max_consecutive_errors` | 4 | Failed tool calls in a row stop the run with `loop_detected` |

A refusal emits a `loop_detected` event and an audit entry, so a sensible agent can recover and a stubborn
one is stopped.

## Token budgets

`Budget(max_tokens_per_run, max_tokens_per_day)`.

- Usage comes from the provider (`usage_metadata`); when a provider reports none it is **estimated** and the
  run's `done` event says how many calls were estimated.
- A run stops (`budget_exceeded`) before the next model call once the limit is reached. A final answer is
  never discarded, so a run can exceed its limit by at most one model call.
- The daily limit (UTC day, all runs) is checked when a run is submitted: `RunRejected("DAILY_BUDGET_EXCEEDED")`.
- Per-run totals are stored in the `runs` table (`input_tokens`, `output_tokens`, `model_calls`).

Budgets count tokens, not money: prices differ per provider and change. Convert with your provider's price list.

## Audit log

Append-only table `audit_log`, readable with `runtime.audit_log(run_id)`:

| kind | When |
| --- | --- |
| `tool_decision` | After policy evaluation, before anything runs. Includes decision, rule id, reason, redacted arguments |
| `approval_approved` / `approval_rejected` / `approval_expired` / `approval_cancelled` | How a request ended |
| `tool_outcome` | After execution: `success`, `error` or `clarification_required` |
| `loop_blocked` / `loop_detected` / `budget_exceeded` | Why a run was refused or stopped |

The decision is written *before* execution, so a crash mid-call still leaves a record of what was intended.
The store does not enforce immutability; protect the database with your platform's controls
(or ship entries to a write-once sink) if you need tamper evidence.

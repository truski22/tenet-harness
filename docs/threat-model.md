# Threat model

What the harness defends against, how, and what it deliberately does **not** cover. Scope: one process
running agents with tools; the model provider and the tools' backends are treated as separate trust zones.

## Assets

- Systems and data the agent's tools can change (tickets, repositories, infrastructure).
- Credentials passed to tools and integrations.
- The integrity of the audit trail and of conversation state.
- Spend (model tokens) and availability of the runtime.

## Trust boundaries

| Source | Trust |
| --- | --- |
| Operator-written agent definitions, policy file, tools | Trusted code and configuration |
| End-user messages | Untrusted input, authenticated by the embedding application |
| Tool results from **external** toolsets (APIs, tickets, merge requests, web pages) | **Untrusted data**, may contain instructions aimed at the model |
| Model output | Untrusted: can request any registered tool with any arguments |

## Threats and mitigations

| # | Threat | Mitigation | Residual risk |
| --- | --- | --- | --- |
| T1 | **Prompt injection through tool results** makes the model take actions the user did not ask for | External results are wrapped as `untrusted_data`, the prompt tells the model to treat them as quoted data, and **writes to external systems need human approval by default**. The approval shows the real arguments | A model can still be misled into *asking* for a harmful action; the safeguard is the human check, not the model. Approvers must read the request |
| T2 | Model calls a tool outside its allow-list | Tools not bound for the turn are refused (`Tool is not permitted for this agent`) | None known for registered tools |
| T3 | Model supplies malformed or hostile arguments | Pydantic validation before policy and execution; policy conditions test the validated values | Tool implementations must still treat their inputs as untrusted |
| T4 | Agent loops and burns tokens or hammers a backend | `LoopPolicy` refuses repeats and stops failure streaks; per-run and per-day token budgets; tool-call budget; per-run deadline | Loops that vary their arguments each time are caught only by the failure streak, the tool-call budget and the deadline |
| T5 | Secrets leak through logs, approvals or the audit trail | `redact` masks credential-like keys before storage and display; logs record exception types, not messages or prompts | Redaction is key-based. A secret inside a free-text value is not detected |
| T6 | A destructive action is repeated after a crash | Open tool calls are closed with "outcome unknown, do not replay"; the tool is never re-executed automatically; transport retries reuse the run ID | An action that completed just before a crash is recorded as unknown, not as done |
| T7 | Two callers approve the same request, or approve after it expired | Atomic single resolution in the store; expired and cancelled requests cannot be approved | None known |
| T8 | A run starves others (resource exhaustion) | `max_concurrent_runs`, one run per conversation, shared `ModelGate`, run and model timeouts | The in-process runtime does not isolate tenants. See below |
| T9 | Policy is weakened by accident | Policy is a reviewed file; first-match semantics are explicit; the external-write baseline needs an explicit `allow` rule to lift | The policy file is trusted. Protect it like code |

## Not covered (by design)

- **Sandboxing of tool code.** Tool allow-lists are code-level boundaries, not a sandbox. Do not load
  untrusted third-party tools; run risky tools in a separate, least-privilege process or account.
- **Authentication and authorization of API callers.** The runtime does not know who a user is. The
  embedding application must authenticate callers, decide who may resolve approvals (`actor` is recorded,
  not verified) and scope data per user.
- **Multi-tenant isolation.** One process, one store. Run one runtime per trust domain.
- **Audit immutability.** The table is append-only by convention. Use database permissions or an external
  write-once sink for tamper evidence.
- **Model-provider risks** (data retention, jailbreaks of the model itself, supply chain of model weights).
- **Durable human-in-the-loop across restarts.** Approvals are held in memory; a restart expires them.

## Evidence

Behaviour above is covered by tests in `tests/integration/test_governance.py`, `tests/unit/test_policy.py` and
`tests/integration/test_store_contract.py`. A downstream application should add its own adversarial cases
to its evaluation dataset (the Second Brain reference app includes prompt-injection cases against a fake
GitLab server) and treat any failure as critical.

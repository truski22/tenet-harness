import pytest
from pydantic import BaseModel

from tenet_harness import Decision, Effect, Policy, PolicyError, Tool, ToolResult, Toolset, default_redact
from tenet_harness.policy import BASELINE_RULE_ID


class Args(BaseModel):
    project: str = ""


def tool(name="gitlab_close", effect=Effect.WRITE):
    return Tool(name, "d", Args, lambda c, a: ToolResult(status="success"), effect)


def toolset(external=True, tool_obj=None):
    return Toolset("gitlab", "GitLab", (tool_obj or tool(),), external=external)


def evaluate(policy, t=None, ts=None, args=None):
    t = t or tool()
    return policy.evaluate(t, ts if ts is not None else toolset(tool_obj=t), args or {})


def test_default_allows_local_writes_and_reads():
    policy = Policy()
    assert evaluate(policy, ts=toolset(external=False)).decision == Decision.ALLOW
    assert evaluate(policy, tool("r", Effect.READ)).decision == Decision.ALLOW


def test_baseline_requires_approval_for_external_writes_only():
    decision = evaluate(Policy())
    assert decision.decision == Decision.REQUIRE_APPROVAL and decision.rule_id == BASELINE_RULE_ID
    destructive = evaluate(Policy(), tool("d", Effect.DESTRUCTIVE))
    assert destructive.decision == Decision.REQUIRE_APPROVAL
    assert evaluate(Policy(), tool("r", Effect.READ)).decision == Decision.ALLOW


def test_first_matching_rule_wins_and_can_override_the_baseline():
    policy = Policy.from_dict({"rules": [
        {"id": "deny-prod", "tools": ["gitlab_*"], "decision": "deny", "reason": "No prod.",
         "when": [{"arg": "project", "op": "matches", "value": "^prod/"}]},
        {"id": "allow-sandbox", "toolsets": ["gitlab"], "decision": "allow",
         "when": [{"arg": "project", "op": "eq", "value": "sandbox/app"}]},
    ]})
    denied = evaluate(policy, args={"project": "prod/api"})
    assert (denied.decision, denied.rule_id, denied.reason) == (Decision.DENY, "deny-prod", "No prod.")
    assert evaluate(policy, args={"project": "sandbox/app"}).decision == Decision.ALLOW  # explicit override
    assert evaluate(policy, args={"project": "other"}).rule_id == BASELINE_RULE_ID


def test_effects_toolsets_and_default():
    policy = Policy.from_dict({"default": "deny", "rules": [
        {"id": "reads", "effects": ["read"], "decision": "allow"}]})
    assert evaluate(policy, tool("r", Effect.READ), toolset(external=False)).decision == Decision.ALLOW
    assert evaluate(policy, ts=toolset(external=False)).decision == Decision.DENY
    other = Policy.from_dict({"rules": [{"id": "x", "toolsets": ["jira"], "decision": "deny"}]})
    assert evaluate(other, ts=toolset(external=False)).decision == Decision.ALLOW


@pytest.mark.parametrize("op, value, have, expected", [
    ("eq", "a", "a", True), ("eq", "a", "b", False), ("ne", "a", "b", True),
    ("in", ["a", "b"], "b", True), ("in", ["a"], "c", False), ("not_in", ["a"], "c", True),
    ("matches", "^pr", "prod", True), ("matches", "^pr", "dev", False),
])
def test_condition_operators(op, value, have, expected):
    policy = Policy.from_dict({"default": "allow", "rules": [
        {"id": "r", "decision": "deny", "when": [{"arg": "project", "op": op, "value": value}]}]})
    result = evaluate(policy, ts=toolset(external=False), args={"project": have})
    assert (result.decision == Decision.DENY) is expected


def test_missing_argument_never_satisfies_a_condition_and_dotted_paths_work():
    policy = Policy.from_dict({"rules": [
        {"id": "r", "decision": "deny", "when": [{"arg": "target.env", "op": "eq", "value": "prod"}]}]})
    ts = toolset(external=False)
    assert evaluate(policy, ts=ts, args={}).decision == Decision.ALLOW
    assert evaluate(policy, ts=ts, args={"target": {"env": "prod"}}).decision == Decision.DENY


def test_yaml_loading_and_validation_errors():
    policy = Policy.from_yaml("""
version: 1
default: allow
rules:
  - id: approve-writes
    effects: [write, destructive]
    decision: require_approval
""")
    assert policy.rules[0].effects == [Effect.WRITE, Effect.DESTRUCTIVE]
    for bad in ("- not a mapping", "rules: [", "rules:\n  - {id: a, decision: maybe}",
                "rules:\n  - {id: a, decision: deny}\n  - {id: a, decision: allow}",
                "rules:\n  - {id: a, decision: deny, when: [{arg: x, op: matches, value: '('}]}",
                "rules:\n  - {id: a, decision: deny, when: [{arg: x, op: in, value: 3}]}",
                "unknown: 1", "rules:\n  - {id: a, decision: deny, surprise: 1}"):
        with pytest.raises(PolicyError):
            Policy.from_yaml(bad)


def test_policy_loads_from_file(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("default: deny\n", encoding="utf-8")
    assert Policy.from_yaml(path).default == Decision.DENY


def test_redaction_masks_credentials_recursively_without_mutating():
    original = {"project": "p", "api_key": "k", "nested": {"Authorization": "Bearer x", "ok": 1},
                "items": [{"password": "pw", "name": "n"}]}
    redacted = default_redact(original)
    assert redacted == {"project": "p", "api_key": "[REDACTED]",
                        "nested": {"Authorization": "[REDACTED]", "ok": 1},
                        "items": [{"password": "[REDACTED]", "name": "n"}]}
    assert original["api_key"] == "k"

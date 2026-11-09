"""Rule-based scoring of one case: tools called, arguments, resulting state, reply text.

No LLM judge: every check is a rule over what the run actually did, so results are reproducible
and comparable between models and code versions. The application supplies the Observation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from ..agents import META_TOOL


@dataclass
class Observation:
    """Everything measurable about one conversation (one or more turns) of a case."""
    status: str                      # status of the last run: completed / failed / ...
    reply: str                       # final reply of the last turn
    tools: list[dict]                # ordered {"name", "status"} from tool_result events
    calls: list[dict]                # ordered {"name", "args"} recorded at tool execution
    evidence: str = ""               # text the reply may legitimately draw numbers from
    state: dict = field(default_factory=dict)  # application state after the run
    seconds: float = 0.0


def _matches(patterns, name: str) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _value_matches(want, have) -> bool:
    if isinstance(want, str) and want.startswith("re:"):
        return re.search(want[3:], str(have), re.I) is not None
    if isinstance(want, str):
        return str(have).casefold() == want.casefold()
    return have == want


def _state_problems(expected: dict, state: dict) -> list[str]:
    problems = []
    for key, want in expected.items():
        have = state.get(key)
        if isinstance(want, dict):
            items = [str(item).casefold() for item in have or []]
            problems += [f"{key} lacks '{needle}'" for needle in want.get("contains", [])
                         if not any(needle.casefold() in item for item in items)]
            problems += [f"{key} unexpectedly has '{needle}'" for needle in want.get("not_contains", [])
                         if any(needle.casefold() in item for item in items)]
        elif have != want:
            problems.append(f"{key} is {have!r}, expected {want!r}")
    return problems


def score(expect: dict, obs: Observation) -> dict:
    """Apply every check the case declares. A run passes only if all declared checks pass."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, why: str):
        checks[name] = ok
        if not ok:
            failures.append(f"{name}: {why}")

    check("completed", obs.status == "completed", f"run ended as '{obs.status}'")
    called = [tool["name"] for tool in obs.tools if tool["name"] != META_TOOL]
    stats: dict = {}

    required = expect.get("tools_required", [])
    forbidden = expect.get("tools_forbidden", [])
    only = expect.get("tools_only")
    if required or forbidden or only is not None:
        missing = [name for name in required if name not in called]
        # A forbidden tool is a violation only if it took effect. An attempt that the application's own
        # guards rejected is tracked separately (blocked_attempts): safe, but the model leaned on the guard.
        hit = [tool["name"] for tool in obs.tools if _matches(forbidden, tool["name"]) and tool["status"] == "success"]
        blocked = [tool["name"] for tool in obs.tools if _matches(forbidden, tool["name"]) and tool["status"] != "success"]
        if forbidden:
            stats["blocked_attempts"] = len(blocked)
        extra = [name for name in called if only is not None and not _matches([*only, *required], name)]
        problems = []
        if missing:
            problems.append(f"did not call {missing}")
        if hit:
            problems.append(f"forbidden tool took effect {hit}")
        if extra:
            problems.append(f"called unneeded {extra}")
        check("tools", not problems, "; ".join(problems) + f" (called {called})")
        if required:
            stats.update(required_total=len(required), required_found=len(required) - len(missing))
        if only is not None:
            stats.update(unneeded_calls=len(extra))

    for spec in expect.get("args", []):
        matching = [call for call in obs.calls if call["name"] == spec["tool"]
                    and all(_value_matches(want, call["args"].get(key)) for key, want in spec["match"].items())]
        check("args", bool(matching), f"no call to {spec['tool']} with {spec['match']} "
                                      f"(calls: {[c for c in obs.calls if c['name'] == spec['tool']]})")

    if "state" in expect:
        problems = _state_problems(expect["state"], obs.state)
        check("state", not problems, "; ".join(problems))

    reply = obs.reply or ""
    rules = expect.get("reply", {})
    if rules:
        problems = []
        if rules.get("any") and not any(re.search(p, reply, re.I) for p in rules["any"]):
            problems.append(f"matches none of {rules['any']}")
        problems += [f"missing /{p}/" for p in rules.get("all", []) if not re.search(p, reply, re.I)]
        problems += [f"forbidden /{p}/ present" for p in rules.get("none", []) if re.search(p, reply, re.I)]
        check("reply", not problems, "; ".join(problems) + f" (reply: {reply[:160]!r})")

    ungrounded: list[str] = []
    if expect.get("grounded"):
        # Numbers of 2+ digits (IDs, counts, years) must come from tools, memories, the user or the clock.
        ungrounded = sorted({n for n in re.findall(r"\d{2,}", reply) if n not in obs.evidence})
        check("grounded", not ungrounded, f"numbers not found in any tool result or context: {ungrounded}")

    return {"passed": all(checks.values()), "checks": checks, "failures": failures, "tools": called,
            "ungrounded": ungrounded, "tool_stats": stats, "calls": obs.calls[:10], "seconds": round(obs.seconds, 2),
            "reply": reply[:2000]}

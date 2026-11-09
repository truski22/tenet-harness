"""Objective, deterministic scoring of agent runs.

No LLM judge: every check is a rule over what the run actually did (tools called, arguments,
resulting application state, reply text), so results are reproducible and comparable between
models and between code versions. This module is application-agnostic; the application supplies
the observation (see ``scripts/eval.py``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from statistics import mean

from .contracts import META_TOOL


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


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a pass rate. Wide on purpose: it shows how little a small suite proves."""
    if total == 0:
        return 0.0, 0.0
    p, d = successes / total, 1 + z * z / total
    centre, margin = p + z * z / (2 * total), z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return round((centre - margin) / d, 3), round((centre + margin) / d, 3)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)], 2)


def summarize(cases: list[dict]) -> dict:
    """``cases``: [{id, category, critical, runs: [score(...) dicts]}] -> metrics."""
    runs = [(case, run) for case in cases for run in case["runs"]]
    passed = sum(run["passed"] for _, run in runs)
    per_case = []
    for case in cases:
        results = [run["passed"] for run in case["runs"]]
        per_case.append({"id": case["id"], "category": case["category"], "critical": case["critical"],
                         "runs": len(results), "passes": sum(results),
                         "pass_rate": round(sum(results) / len(results), 3) if results else 0.0,
                         "pass_all": bool(results) and all(results),
                         "failures": sorted({f for run in case["runs"] for f in run["failures"]})})
    blocked_runs = sum(run["tool_stats"].get("blocked_attempts", 0) > 0 for _, run in runs)
    found = sum(run["tool_stats"].get("required_found", 0) for _, run in runs)
    total = sum(run["tool_stats"].get("required_total", 0) for _, run in runs)
    only_runs = [run for _, run in runs if "unneeded_calls" in run["tool_stats"]]
    categories = {}
    for name in sorted({case["category"] for case in cases}):
        rows = [run["passed"] for case, run in runs if case["category"] == name]
        categories[name] = {"runs": len(rows), "pass_at_1": round(sum(rows) / len(rows), 3)}
    seconds = [run["seconds"] for _, run in runs]
    return {
        "cases": len(cases), "runs": len(runs),
        "pass_at_1": round(passed / len(runs), 3) if runs else 0.0,
        "pass_at_1_ci95": wilson(passed, len(runs)),
        # pass^k: share of cases that succeed in EVERY repetition; the reliability number.
        "pass_all_k": round(sum(c["pass_all"] for c in per_case) / len(per_case), 3) if per_case else 0.0,
        "critical_failures": sum(not run["passed"] for case, run in runs if case["critical"]),
        "tool_recall": round(found / total, 3) if total else None,
        "unneeded_tool_call_runs": round(
            sum(run["tool_stats"]["unneeded_calls"] > 0 for run in only_runs) / len(only_runs), 3
        ) if only_runs else None,
        "blocked_attempt_runs": blocked_runs,
        "ungrounded_runs": sum(bool(run["ungrounded"]) for _, run in runs),
        "latency_seconds": {"mean": round(mean(seconds), 2) if seconds else 0.0,
                            "p50": percentile(seconds, .5), "p95": percentile(seconds, .95)},
        "by_category": categories, "per_case": per_case,
    }


def compare(current: dict, baseline: dict, tolerance: float = 0.05) -> dict:
    """Case-level diff of two summaries (same dataset assumed; see ``same_dataset``)."""
    now = {c["id"]: c for c in current["summary"]["per_case"]}
    was = {c["id"]: c for c in baseline["summary"]["per_case"]}
    shared = now.keys() & was.keys()
    regressions = [{"id": i, "was": was[i]["pass_rate"], "now": now[i]["pass_rate"]}
                   for i in sorted(shared) if now[i]["pass_rate"] < was[i]["pass_rate"]]
    improvements = [{"id": i, "was": was[i]["pass_rate"], "now": now[i]["pass_rate"]}
                    for i in sorted(shared) if now[i]["pass_rate"] > was[i]["pass_rate"]]
    delta = round(current["summary"]["pass_at_1"] - baseline["summary"]["pass_at_1"], 3)
    reasons = []
    if current["summary"]["critical_failures"]:
        reasons.append(f"{current['summary']['critical_failures']} critical run(s) failed")
    if delta < -tolerance:
        reasons.append(f"pass@1 dropped {abs(delta):.3f} (tolerance {tolerance})")
    return {"baseline_model": baseline["meta"]["model"], "pass_at_1_delta": delta,
            "pass_all_k_delta": round(current["summary"]["pass_all_k"] - baseline["summary"]["pass_all_k"], 3),
            "regressions": regressions, "improvements": improvements,
            "same_dataset": current["meta"]["cases_sha256"] == baseline["meta"]["cases_sha256"],
            "new_cases": sorted(now.keys() - was.keys()), "removed_cases": sorted(was.keys() - now.keys()),
            "gate": {"ok": not reasons, "reasons": reasons}}


def render_markdown(report: dict, comparison: dict | None = None) -> str:
    s, m = report["summary"], report["meta"]
    low, high = s["pass_at_1_ci95"]
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
    lines = [f"# Evaluation: {m['model']}", "",
             f"{s['cases']} cases x {m['repeat']} repetitions = {s['runs']} runs, "
             f"{m['duration_seconds']:.0f}s, dataset `{m['cases_sha256'][:10]}`", "",
             "| Metric | Value |", "| --- | --- |",
             f"| pass@1 (average run) | {pct(s['pass_at_1'])} (95% CI {pct(low)} - {pct(high)}) |",
             f"| pass^k (case passes every repetition) | {pct(s['pass_all_k'])} |",
             f"| Critical-case failures (must be 0) | {s['critical_failures']} |",
             f"| Tool recall (required tools called) | {pct(s['tool_recall'])} |",
             f"| Runs with unneeded tool calls | {pct(s['unneeded_tool_call_runs'])} |",
             f"| Runs where a guard blocked a forbidden tool attempt | {s['blocked_attempt_runs']} |",
             f"| Runs with ungrounded numbers | {s['ungrounded_runs']} |",
             f"| Latency mean / p50 / p95 | {s['latency_seconds']['mean']}s / "
             f"{s['latency_seconds']['p50']}s / {s['latency_seconds']['p95']}s |", "",
             "| Category | Runs | pass@1 |", "| --- | --- | --- |"]
    lines += [f"| {name} | {row['runs']} | {pct(row['pass_at_1'])} |" for name, row in s["by_category"].items()]
    failing = [c for c in s["per_case"] if not c["pass_all"]]
    if failing:
        lines += ["", "## Cases that did not pass every repetition", ""]
        for case in failing:
            flag = " **[critical]**" if case["critical"] else ""
            lines.append(f"- `{case['id']}`{flag}: {case['passes']}/{case['runs']}")
            lines += [f"    - {failure}" for failure in case["failures"][:3]]
    if comparison:
        lines += ["", f"## Versus baseline ({comparison['baseline_model']})", "",
                  f"pass@1 delta {comparison['pass_at_1_delta']:+.3f}, "
                  f"pass^k delta {comparison['pass_all_k_delta']:+.3f}"]
        if not comparison["same_dataset"]:
            lines.append("- WARNING: dataset differs from the baseline; deltas are not like-for-like.")
        lines += [f"- regression `{r['id']}`: {r['was']:.2f} -> {r['now']:.2f}" for r in comparison["regressions"]]
        lines += [f"- improvement `{r['id']}`: {r['was']:.2f} -> {r['now']:.2f}" for r in comparison["improvements"]]
        lines.append("- gate: " + ("OK" if comparison["gate"]["ok"] else "FAILED - " + "; ".join(comparison["gate"]["reasons"])))
    return "\n".join(lines) + "\n"

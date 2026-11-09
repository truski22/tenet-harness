"""Aggregate case results, compare against a baseline and render a Markdown report."""
from __future__ import annotations

from statistics import mean

from .stats import percentile, wilson


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

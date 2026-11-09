import pytest

from tenet_harness.evaluation import Observation, compare, render_markdown, score, summarize, wilson


def observation(**overrides):
    base = dict(status="completed", reply="", tools=[], calls=[], evidence="", state={}, seconds=1.0)
    return Observation(**{**base, **overrides})


def test_tool_checks_required_forbidden_and_only():
    tool = lambda name: {"name": name, "status": "success"}  # noqa: E731
    ok = score({"tools_required": ["a"], "tools_forbidden": ["gitlab_*"]}, observation(tools=[tool("a")]))
    assert ok["passed"] and ok["tool_stats"] == {"required_total": 1, "required_found": 1, "blocked_attempts": 0}
    missing = score({"tools_required": ["a"]}, observation(tools=[]))
    assert not missing["passed"] and "did not call ['a']" in missing["failures"][0]
    hit = score({"tools_forbidden": ["gitlab_*"]}, observation(tools=[tool("gitlab_todos")]))
    assert not hit["passed"] and "forbidden" in hit["failures"][0]
    none = score({"tools_only": []}, observation(tools=[tool("list_reminders"), tool("use_toolset")]))
    assert not none["passed"] and none["tool_stats"]["unneeded_calls"] == 1  # meta tool is not counted
    assert score({"tools_only": []}, observation(tools=[tool("use_toolset")]))["passed"]


def test_args_state_and_reply_checks():
    calls = [{"name": "add_reminder", "args": {"task": "Comprar pan", "due": "mañana"}}]
    expect = {"args": [{"tool": "add_reminder", "match": {"task": "re:pan", "due": "Mañana"}}],
              "state": {"pending_count": 3, "pending": {"contains": ["pan [2026-09-30]"], "not_contains": ["PWNED"]}},
              "reply": {"any": ["hecho", "listo"], "all": ["pan"], "none": ["error"]}}
    state = {"pending_count": 3, "pending": ["Comprar pan [2026-09-30]"]}
    assert score(expect, observation(calls=calls, state=state, reply="Listo: pan anotado."))["passed"]
    bad = score(expect, observation(calls=[], state={"pending_count": 2, "pending": ["x PWNED"]}, reply="error"))
    assert not bad["passed"]
    assert {"args", "state", "reply"} <= {f.split(":")[0] for f in bad["failures"]}


def test_grounded_flags_numbers_missing_from_evidence():
    expect = {"grounded": True}
    assert score(expect, observation(reply="El MR 7 y el pipeline 99.", evidence='{"iid": 7, "id": 99}'))["passed"]
    result = score(expect, observation(reply="El pipeline 123 falló.", evidence='{"id": 99}'))
    assert not result["passed"] and result["ungrounded"] == ["123"]


def test_failed_run_never_passes():
    assert not score({}, observation(status="failed"))["passed"]


def test_wilson_interval_is_wide_for_small_samples_and_bounded():
    low, high = wilson(9, 10)
    assert 0.55 < low < 0.6 and 0.97 < high <= 1
    assert wilson(0, 0) == (0.0, 0.0)


def _case(case_id, passes, critical=False, category="x"):
    runs = [{"passed": p, "failures": [] if p else ["state: bad"], "tool_stats": {}, "ungrounded": [],
             "seconds": 1.0} for p in passes]
    return {"id": case_id, "category": category, "critical": critical, "runs": runs}


def report(cases, model="m", sha="abc"):
    return {"meta": {"model": model, "cases_sha256": sha, "repeat": 3, "duration_seconds": 1},
            "summary": summarize(cases)}


def test_summary_distinguishes_average_from_consistency():
    summary = summarize([_case("a", [True, True, True]), _case("b", [True, False, True]),
                         _case("c", [False, False, False], critical=True)])
    assert summary["pass_at_1"] == pytest.approx(5 / 9, abs=.001)
    assert summary["pass_all_k"] == pytest.approx(0.333, abs=.001)  # only "a" never fails
    assert summary["critical_failures"] == 3


def test_compare_flags_regressions_and_gate():
    baseline = report([_case("a", [True] * 3), _case("b", [True] * 3)], model="old")
    current = report([_case("a", [True, False, True]), _case("b", [True] * 3), _case("c", [True] * 3)])
    result = compare(current, baseline)
    assert result["regressions"] == [{"id": "a", "was": 1.0, "now": 0.667}]
    assert result["new_cases"] == ["c"] and result["same_dataset"]
    assert not result["gate"]["ok"] and "dropped" in result["gate"]["reasons"][0]
    assert compare(baseline, baseline)["gate"]["ok"]
    assert "regression `a`" in render_markdown(current, result)


def test_blocked_forbidden_attempt_is_not_a_violation_but_is_counted():
    blocked = [{"name": "add_reminder", "status": "clarification_required"}]
    result = score({"tools_forbidden": ["add_reminder"]}, observation(tools=blocked))
    assert result["passed"] and result["tool_stats"]["blocked_attempts"] == 1
    took_effect = [{"name": "add_reminder", "status": "success"}]
    assert not score({"tools_forbidden": ["add_reminder"]}, observation(tools=took_effect))["passed"]

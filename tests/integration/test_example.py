import runpy
from pathlib import Path


def test_echo_example_runs_end_to_end(capsys):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "echo_agent.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "The tool returned: hello" in out and "('tool_result', 'echo')" in out


def test_approval_example_runs_end_to_end(capsys):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "approval_agent.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "Approval needed: close_ticket" in out and "[REDACTED]" in out and "closed: ['OPS-42']" in out


def test_example_policy_file_is_valid():
    from tenet_harness import Policy
    policy = Policy.from_yaml(Path(__file__).resolve().parents[2] / "examples" / "policy.yaml")
    assert [rule.id for rule in policy.rules] == ["production-is-read-only", "sandbox-is-free",
                                                  "destructive-needs-a-human"]

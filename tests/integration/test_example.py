import runpy
from pathlib import Path


def test_echo_example_runs_end_to_end(capsys):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples" / "echo_agent.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "The tool returned: hello" in out and "('tool_result', 'echo')" in out

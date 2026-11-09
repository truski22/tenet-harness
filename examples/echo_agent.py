"""Smallest complete agent: one read-only tool, scripted model, no Ollama or network needed.

    uv run python examples/echo_agent.py
"""
import tempfile
import time
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

from tenet_harness.contracts import AgentDefinition, Effect, Registry, RunRequest, Tool, ToolResult, Toolset
from tenet_harness.runtime import Runtime, TERMINAL
from tenet_harness.storage import Store
from tenet_harness.testing import FakeProvider


class EchoInput(BaseModel):
    text: str


echo = Tool("echo", "Return the supplied text without side effects.", EchoInput,
            lambda context, args: ToolResult(status="success", data=args.text), Effect.READ)

agent = AgentDefinition(
    id="echo", name="Echo", description="Isolated example agent.",
    toolsets=(Toolset("echo", "Echo text back.", (echo,), always_on=True),),
    context=lambda context: SystemMessage(content="Use only your registered tools."),
)

if __name__ == "__main__":
    registry = Registry()
    registry.register(agent)
    # The scripted model first calls the tool, then answers; swap FakeProvider for OllamaProvider to go live.
    model = FakeProvider([
        AIMessage(content="", tool_calls=[{"name": "echo", "id": "c1", "args": {"text": "hello"}}]),
        AIMessage(content="The tool returned: hello"),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "harness.sqlite")
        runtime = Runtime(registry, model, store)
        runtime.submit(RunRequest(run_id="r1", thread_id="t1", agent_id="echo", text="Say hello"))
        while store.run("r1")["status"] not in TERMINAL:
            time.sleep(0.05)
        print(store.run("r1")["reply"])
        print([(e["type"], e.get("name")) for e in store.events("r1")])
        runtime.close()

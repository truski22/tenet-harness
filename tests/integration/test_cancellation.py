import threading
import time

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from tenet_harness.contracts import AgentDefinition, Registry, RunRequest
from tenet_harness.runtime import BusyError, Runtime
from tenet_harness.storage import Store


def test_busy_and_cancellation_preserve_terminal_state(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Model:
        def invoke(self, messages):
            entered.set()
            release.wait(5)
            return AIMessage(content="Should not be displayed after cancellation")

    class Provider:
        def chat(self, structured=False):
            return Model()

    registry = Registry()
    registry.register(AgentDefinition("a", "A", "", (), lambda ctx: SystemMessage(content="Test.")))
    runtime = Runtime(registry, Provider(), Store(tmp_path / "test.sqlite"))
    try:
        request = RunRequest(run_id="r", thread_id="t", agent_id="a", text="hello")
        runtime.submit(request)
        assert entered.wait(5)
        with pytest.raises(BusyError):
            runtime.submit(request.model_copy(update={"run_id": "r2"}))
        runtime.cancel("r")
        release.set()
        runtime.run_thread.join(5)
        assert runtime.store.run("r")["status"] == "cancelled"
        assert not any(e["type"] == "reply" for e in runtime.store.events("r"))
    finally:
        release.set()
        runtime.close()


def test_run_deadline(tmp_path):
    class Provider:
        def chat(self, structured=False):
            return self

        def invoke(self, messages):
            time.sleep(.05)
            return AIMessage(content="Late")
    registry = Registry()
    registry.register(AgentDefinition("a", "A", "", (), lambda ctx: SystemMessage(content="Test.")))
    runtime = Runtime(registry, Provider(), Store(tmp_path / "test.sqlite"), run_timeout=.01)
    try:
        runtime.submit(RunRequest(run_id="r", thread_id="t", agent_id="a", text="hello"))
        runtime.run_thread.join(5)
        assert runtime.store.run("r")["status"] == "failed"
    finally:
        runtime.close()

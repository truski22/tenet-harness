import threading
import time

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from tenet_harness import AgentDefinition, BusyError, Registry, RunRequest, Runtime, Store
from tenet_harness.runtime import ModelGate


class BlockingProvider:
    """Model whose calls wait on an event; records how many ran at the same time."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def chat(self, structured=False):
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.entered.release()
        try:
            assert self.release.wait(10)
            return AIMessage(content="done")
        finally:
            with self.lock:
                self.running -= 1


def make_runtime(tmp_path, provider, **options):
    registry = Registry()
    registry.register(AgentDefinition("a", "A", "", (), lambda ctx: SystemMessage(content="Test.")))
    return Runtime(registry, provider, Store(tmp_path / "t.sqlite"), **options)


def request(run_id, thread_id):
    return RunRequest(run_id=run_id, thread_id=thread_id, agent_id="a", text="hello")


def test_runs_execute_concurrently_up_to_the_limit(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider, max_concurrent_runs=2, model_concurrency=2)
    try:
        runtime.submit(request("r1", "t1"))
        runtime.submit(request("r2", "t2"))
        assert provider.entered.acquire(timeout=5) and provider.entered.acquire(timeout=5)
        assert set(runtime.active_run_ids) == {"r1", "r2"}
        with pytest.raises(BusyError):
            runtime.submit(request("r3", "t3"))
        provider.release.set()
        assert runtime.wait("r1", 5) and runtime.wait("r2", 5)
        assert provider.peak == 2
        assert {runtime.store.run(r)["status"] for r in ("r1", "r2")} == {"completed"}
    finally:
        provider.release.set()
        runtime.close()


def test_one_active_run_per_conversation(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider, max_concurrent_runs=3, model_concurrency=3)
    try:
        runtime.submit(request("r1", "same"))
        assert provider.entered.acquire(timeout=5)
        with pytest.raises(BusyError):
            runtime.submit(request("r2", "same"))
    finally:
        provider.release.set()
        runtime.close()


def test_model_concurrency_serializes_model_calls(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider, max_concurrent_runs=2, model_concurrency=1)
    try:
        runtime.submit(request("r1", "t1"))
        runtime.submit(request("r2", "t2"))
        assert provider.entered.acquire(timeout=5)
        time.sleep(.3)  # the second run must be waiting for the gate, not inside the model
        assert provider.running == 1
        provider.release.set()
        assert runtime.wait("r1", 5) and runtime.wait("r2", 5)
        assert provider.peak == 1
    finally:
        provider.release.set()
        runtime.close()


def test_cancelling_one_run_leaves_the_other_running(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider, max_concurrent_runs=2, model_concurrency=2)
    try:
        runtime.submit(request("r1", "t1"))
        runtime.submit(request("r2", "t2"))
        assert provider.entered.acquire(timeout=5) and provider.entered.acquire(timeout=5)
        runtime.cancel("r1")
        provider.release.set()
        assert runtime.wait("r1", 5) and runtime.wait("r2", 5)
        assert runtime.store.run("r1")["status"] == "cancelled"
        assert runtime.store.run("r2")["status"] == "completed"
    finally:
        provider.release.set()
        runtime.close()


def test_exclusive_requires_idle_runtime_and_blocks_submissions(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider)
    try:
        with runtime.exclusive():
            with pytest.raises(BusyError):
                runtime.submit(request("r1", "t1"))
            with pytest.raises(BusyError):
                with runtime.exclusive():
                    pass
        runtime.submit(request("r1", "t1"))
        assert provider.entered.acquire(timeout=5)
        with pytest.raises(BusyError):
            with runtime.exclusive():
                pass
    finally:
        provider.release.set()
        runtime.close()


def test_close_cancels_active_runs(tmp_path):
    provider = BlockingProvider()
    runtime = make_runtime(tmp_path, provider)
    runtime.submit(request("r1", "t1"))
    assert provider.entered.acquire(timeout=5)
    provider.release.set()
    runtime.close()
    assert runtime.store.run("r1")["status"] in {"completed", "cancelled"}


def test_model_gate_limits_and_drains():
    gate = ModelGate(1)
    inside = threading.Event()
    leave = threading.Event()

    def hold():
        with gate.slot():
            inside.set()
            leave.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert inside.wait(5)
    seen = []

    def second():
        with gate.slot():
            seen.append("second")

    waiter = threading.Thread(target=second)
    waiter.start()
    time.sleep(.3)
    assert seen == []  # blocked while the first holds the only slot
    leave.set()
    thread.join(5)
    waiter.join(5)
    assert seen == ["second"]
    with gate.drain():
        pass


def test_model_gate_wait_is_abortable():
    gate = ModelGate(1)

    def abort():
        raise RuntimeError("abort")

    with gate.slot():
        with pytest.raises(RuntimeError):
            with gate.slot(abort):
                pass


def test_model_gate_rejects_invalid_limit():
    with pytest.raises(ValueError):
        ModelGate(0)

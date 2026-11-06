"""Idle-boundary restart: request file, drain-then-stop, timeout and ETA notice."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from nanobot.gateway.idle_restart import (
    clear_request,
    read_request,
    request_path,
    restart_retry_after_seconds,
    set_restart_retry_after,
    watch_idle_restart,
    write_request,
)
from nanobot.runtime_admission import RuntimeBusyError


@pytest.fixture(autouse=True)
def _clean_retry_state():
    set_restart_retry_after(None)
    yield
    set_restart_retry_after(None)


class _Maintenance:
    """Stand-in for AgentLoop.maintenance: busy N times, then lets intake close."""

    def __init__(self, busy_times: int = 0, error: Exception | None = None) -> None:
        self.busy_times = busy_times
        self.calls = 0
        self.entered = 0
        self.exited = 0
        self.error = error

    @contextmanager
    def _ctx(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.calls <= self.busy_times:
            raise RuntimeBusyError("nanobot has pending work")
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1

    def __call__(self):
        return self._ctx()


async def _until(condition, *, timeout: float = 3.0) -> None:
    """Poll a condition until it holds, so tests never depend on tick counts."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


async def _stop(watcher: asyncio.Task) -> None:
    """The watcher is a forever task: cancel it the way gateway teardown does."""
    watcher.cancel()
    with suppress(asyncio.CancelledError):
        await watcher


def test_request_roundtrip_increments_attempt_and_clamps_eta(tmp_path: Path) -> None:
    path = request_path(tmp_path)
    assert read_request(path) is None

    first = write_request(path, reason="port change", eta_seconds=4)
    assert first.attempt == 1
    loaded = read_request(path)
    assert loaded is not None and loaded.reason == "port change" and loaded.eta_seconds == 4.0

    second = write_request(path, reason="again", eta_seconds=10_000)
    assert second.attempt == 2
    assert read_request(path).eta_seconds == 3600.0  # 上限钳制

    clear_request(path)
    assert read_request(path) is None


def test_read_request_ignores_garbage_and_stale_shapes(tmp_path: Path) -> None:
    path = request_path(tmp_path)
    path.write_text("not json", encoding="utf-8")
    assert read_request(path) is None
    path.write_text('{"attempt": 0}', encoding="utf-8")
    assert read_request(path) is None
    path.write_text("[]", encoding="utf-8")
    assert read_request(path) is None


@pytest.mark.asyncio
async def test_watcher_waits_for_idle_then_holds_the_gate(tmp_path: Path) -> None:
    path = request_path(tmp_path)
    write_request(path, reason="sandbox change", eta_seconds=6)
    maintenance = _Maintenance(busy_times=2)
    shutdown_event = asyncio.Event()
    armed: list[float] = []
    watcher = asyncio.create_task(watch_idle_restart(
        path,
        maintenance=maintenance,
        shutdown_event=shutdown_event,
        poll_s=0.01,
        max_wait_s=5,
        on_arm=lambda request: (armed.append(request.eta_seconds), set_restart_retry_after(request.eta_seconds)),
    ))

    await asyncio.wait_for(shutdown_event.wait(), timeout=3)
    await asyncio.sleep(0.05)

    assert armed == [6.0]
    assert maintenance.calls == 3 and maintenance.entered == 1
    assert read_request(path) is None
    # 排空后仍持有上下文：进程退出前不能重开 intake
    assert maintenance.exited == 0
    assert restart_retry_after_seconds() == 6.0, "ETA 通知要留在整个排空期间"

    await _stop(watcher)
    # 取消任务也不会释放：排空状态一直保持到进程退出，关闭期间不会有新任务被接走
    assert maintenance.exited == 0


@pytest.mark.asyncio
async def test_watcher_gives_up_after_max_wait(tmp_path: Path) -> None:
    path = request_path(tmp_path)
    write_request(path, reason="busy forever")
    maintenance = _Maintenance(busy_times=10_000)
    shutdown_event = asyncio.Event()
    aborted: list[str] = []
    set_restart_retry_after(8)

    watcher = asyncio.create_task(watch_idle_restart(
        path,
        maintenance=maintenance,
        shutdown_event=shutdown_event,
        poll_s=0.01,
        max_wait_s=0.2,
        on_abort=lambda detail: (aborted.append(detail), set_restart_retry_after(None)),
    ))

    await _until(lambda: bool(aborted))
    await _stop(watcher)

    assert shutdown_event.is_set() is False
    assert "gave up" in aborted[0]
    assert read_request(path) is None
    assert restart_retry_after_seconds() is None, "放弃后不能继续对外宣称要重启"


@pytest.mark.asyncio
async def test_watcher_backs_off_when_an_updater_owns_the_gate(tmp_path: Path) -> None:
    path = request_path(tmp_path)
    write_request(path, reason="race with updater")
    maintenance = _Maintenance(error=RuntimeError("nanobot is preparing an update"))
    shutdown_event = asyncio.Event()

    watcher = asyncio.create_task(watch_idle_restart(
        path,
        maintenance=maintenance,
        shutdown_event=shutdown_event,
        poll_s=0.01,
        max_wait_s=2,
    ))

    await _until(lambda: read_request(path) is None)
    await _stop(watcher)

    assert maintenance.calls == 1
    assert shutdown_event.is_set() is False

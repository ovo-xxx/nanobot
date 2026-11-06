"""Restart the gateway at an idle boundary instead of in the middle of a turn.

Needing a restart is normal (bound ports, sandbox policy, channel wiring); interrupting a
running turn is not. The updater already owns the drain primitive
``RuntimeAdmission.pause_if_idle`` - it *refuses* to close intake while work is
outstanding. This adds the missing half: wait for the idle moment instead of failing fast,
announce an ETA so clients stop burning their short reconnect intervals against a port
that is going away, then hand the process to the supervisor.

The request is a small file under the instance's ``run/`` directory so any local operator
(CLI, WebUI action, a script) can ask for it without a network surface, and a crashed
requester cannot leave the gateway armed forever: the watcher only trusts a fresh
``attempt`` number and clears the file once it acts.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.runtime_admission import RuntimeBusyError

REQUEST_FILENAME = "restart-request.json"
DEFAULT_MAX_WAIT_S = 600.0
DEFAULT_ETA_S = 8.0
_POLL_S = 0.5


@dataclass(frozen=True, slots=True)
class RestartRequest:
    """One request to restart when the gateway next goes idle."""

    attempt: int
    reason: str = ""
    eta_seconds: float = DEFAULT_ETA_S
    requested_at: float = 0.0


def request_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / REQUEST_FILENAME


def read_request(path: Path | str) -> RestartRequest | None:
    """Return the pending request, or None when there is none or it is unreadable."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        attempt = int(raw.get("attempt", 0))
    except (TypeError, ValueError):
        return None
    if attempt <= 0:
        return None
    try:
        eta = float(raw.get("eta_seconds", DEFAULT_ETA_S))
    except (TypeError, ValueError):
        eta = DEFAULT_ETA_S
    return RestartRequest(
        attempt=attempt,
        reason=str(raw.get("reason", ""))[:200],
        eta_seconds=max(1.0, min(eta, 3600.0)),
        requested_at=float(raw.get("requested_at") or 0.0),
    )


def write_request(path: Path | str, *, reason: str = "", eta_seconds: float = DEFAULT_ETA_S) -> RestartRequest:
    """Queue a restart, bumping the attempt counter so a repeat request is honoured."""
    target = Path(path)
    previous = read_request(target)
    request = RestartRequest(
        attempt=(previous.attempt + 1) if previous else 1,
        reason=reason,
        eta_seconds=eta_seconds,
        requested_at=time.time(),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(request), ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return request


def clear_request(path: Path | str) -> None:
    with suppress(OSError):
        Path(path).unlink()


# Process-wide drain notice. The gateway is one process and the only writer is the
# restart watcher below, so a module-level value is enough to let the HTTP layer answer
# `503 + Retry-After` during the drain window without threading a new dependency through
# GatewayServices -> GatewayHTTPHandler. It is cleared as soon as the drain ends.
_retry_after_seconds: float | None = None


def set_restart_retry_after(seconds: float | None) -> None:
    global _retry_after_seconds
    _retry_after_seconds = None if seconds is None else max(1.0, min(float(seconds), 3600.0))


def restart_retry_after_seconds() -> float | None:
    return _retry_after_seconds


async def watch_idle_restart(
    path: Path | str,
    *,
    maintenance: Callable[[], Any],
    shutdown_event: asyncio.Event,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    poll_s: float = _POLL_S,
    on_arm: Callable[[RestartRequest], None] | None = None,
    on_abort: Callable[[str], None] | None = None,
) -> None:
    """Wait for an idle moment, then close intake and ask the supervisor to bring us back.

    ``maintenance`` is ``AgentLoop.maintenance``: entering it means nothing new will be
    admitted, and it raises ``RuntimeBusyError`` while work is outstanding. We never
    leave the context - the process is about to stop, and reopening intake mid-shutdown
    would let a fresh turn in.
    """
    seen: int | None = None
    while not shutdown_event.is_set():
        await asyncio.sleep(poll_s)
        request = read_request(path)
        if request is None or request.attempt == seen:
            continue
        seen = request.attempt
        started = time.monotonic()
        logger.info(
            "Idle restart requested (attempt {}, reason {!r}); waiting up to {:.0f}s for work to drain",
            request.attempt, request.reason, max_wait_s,
        )
        if on_arm is not None:
            on_arm(request)

        while not shutdown_event.is_set():
            if time.monotonic() - started > max_wait_s:
                clear_request(path)
                set_restart_retry_after(None)
                detail = f"waited {max_wait_s:.0f}s and gave up"
                logger.warning("Idle restart aborted (attempt {}): still busy after {}", request.attempt, detail)
                if on_abort is not None:
                    on_abort(detail)
                break
            try:
                context = maintenance()
                context.__enter__()
            except RuntimeBusyError:
                await asyncio.sleep(poll_s)
                continue
            except Exception as exc:
                # Another maintenance window owns the gate (an updater in flight);
                # do not fight it, and do not leave a stale request behind.
                clear_request(path)
                set_restart_retry_after(None)
                logger.warning("Idle restart aborted (attempt {}): {}", request.attempt, exc)
                if on_abort is not None:
                    on_abort(str(exc))
                break

            # 上下文已持有（上面 __enter__ 成功）。这里绝不能再进一次，也不能离开：
            # 提前退出会重开 intake，关闭期间的消息就可能被接走；`held` 让协程帧一直
            # 引用它，避免 contextmanager 对象被回收时顺手把生成器关掉。
            held = context  # noqa: F841 - 一直引用，防止生成器被回收时关掉排空状态
            clear_request(path)
            logger.info(
                "Gateway idle after {:.1f}s; shutting down for restart (attempt {}, intake stays closed)",
                time.monotonic() - started, request.attempt,
            )
            shutdown_event.set()
            while True:  # 停在排空状态里，直到 gateway teardown 取消本任务
                await asyncio.sleep(1.0)

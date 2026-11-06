"""Planned-restart notice: bootstrap answers 503 + Retry-After only while draining."""

from __future__ import annotations

import pytest

from nanobot.gateway.idle_restart import set_restart_retry_after
from nanobot.webui.ws_http import restart_pending_response


@pytest.fixture(autouse=True)
def _reset():
    set_restart_retry_after(None)
    yield
    set_restart_retry_after(None)


def test_no_notice_outside_the_drain_window() -> None:
    assert restart_pending_response() is None


def test_notice_carries_the_announced_delay() -> None:
    set_restart_retry_after(8.4)
    response = restart_pending_response()
    assert response is not None
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "8"

"""AgentLoop config knobs: live config wins over boot value, runtime override wins over both."""

from __future__ import annotations

from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import AgentDefaults, GatewayConfig


def _loop(defaults: AgentDefaults, gateway: GatewayConfig) -> AgentLoop:
    """Build just enough of a loop to exercise the knob properties."""
    loop = AgentLoop.__new__(AgentLoop)
    loop._defaults_ref = defaults
    loop._gateway_ref = gateway
    loop._runtime_overrides = {}
    return loop


def test_knobs_follow_the_live_config_objects_that_reload_mutates() -> None:
    defaults = AgentDefaults(max_tool_result_chars=16_000, provider_retry_mode="standard", unified_session=False)
    gateway = GatewayConfig(restart_mode="auto")
    loop = _loop(defaults, gateway)

    assert loop.max_tool_result_chars == 16_000
    assert loop.provider_retry_mode == "standard"
    assert loop.unified_session is False
    assert loop.restart_mode == "auto"

    # 模拟热重载：watcher 原地改这两个对象，不替换指针
    defaults.max_tool_result_chars = 4_000
    defaults.provider_retry_mode = "off"
    defaults.unified_session = True
    gateway.restart_mode = "exit"

    assert loop.max_tool_result_chars == 4_000
    assert loop.provider_retry_mode == "off"
    assert loop.unified_session is True
    assert loop.restart_mode == "exit"


def test_runtime_override_beats_a_later_config_change() -> None:
    """`my set` 是操作者当场动过的旋钮，热重载不该把它抹掉。"""
    defaults = AgentDefaults(max_tool_result_chars=16_000)
    loop = _loop(defaults, GatewayConfig())

    loop.max_tool_result_chars = 900
    defaults.max_tool_result_chars = 5_000

    assert loop.max_tool_result_chars == 900


def test_loop_without_init_falls_back_to_schema_defaults() -> None:
    """若干测试用 AgentLoop.__new__ 构造，读旋钮不能炸。"""
    loop = AgentLoop.__new__(AgentLoop)

    assert loop.max_tool_result_chars == 16_000
    assert loop.provider_retry_mode == "standard"
    assert loop.unified_session is False
    assert loop.restart_mode == "auto"

    loop.unified_session = True
    assert loop.unified_session is True

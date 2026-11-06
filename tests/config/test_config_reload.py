"""Hot config reload: in-place mutation, identity preservation, restart-required refusal."""

from __future__ import annotations

from nanobot.config.reload import apply_config_in_place, is_frozen
from nanobot.config.schema import Config


def _config(**sections: object) -> Config:
    return Config.model_validate(sections)


def test_cheap_scalar_is_applied() -> None:
    live = _config()
    result = apply_config_in_place(live, _config(agents={"defaults": {"max_tool_iterations": 42}}))

    assert "agents.defaults.max_tool_iterations" in result.applied
    assert live.agents.defaults.max_tool_iterations == 42
    assert result.needs_restart is False


def test_nested_identity_survives_so_sub_model_holders_are_not_left_behind() -> None:
    """`AgentLoop.tools_config` / `GatewayHTTPHandler.config` hold sub-objects, not the root."""
    live = _config()
    tools_before = live.tools
    cli_before = live.tools.cli_apps
    defaults_before = live.agents.defaults

    apply_config_in_place(
        live,
        _config(agents={"defaults": {"session_ttl_minutes": 33}}, tools={"cli_apps": {"run_timeout": 120}}),
    )

    assert live.tools is tools_before, "换指针会让持有子对象的消费者读到旧值"
    assert live.tools.cli_apps is cli_before
    assert live.agents.defaults is defaults_before
    assert live.agents.defaults.session_ttl_minutes == 33
    assert live.tools.cli_apps.run_timeout == 120


def test_security_and_startup_only_fields_are_refused_not_mutated() -> None:
    live = _config(
        gateway={"host": "127.0.0.1", "port": 18799},
        channels={"websocket": {"enabled": True, "port": 18900}},
        tools={
            "exec": {"enable": True, "sandbox": "bwrap"},
            "my": {"enable": True, "allow_set": False},
            "restrict_to_workspace": True,
        },
        agents={"defaults": {"workspace": "/root/.nanobot/workspace"}},
    )
    before = (
        live.gateway.port,
        live.channels.websocket["port"],
        live.tools.exec.sandbox,
        live.tools.exec.enable,
        live.tools.my.allow_set,
        live.tools.restrict_to_workspace,
        live.agents.defaults.workspace,
    )

    fresh = _config(
        gateway={"host": "127.0.0.1", "port": 9999},
        channels={"websocket": {"enabled": True, "port": 9999}},
        tools={
            "exec": {"enable": False, "sandbox": "off"},
            "my": {"enable": True, "allow_set": True},
            "restrict_to_workspace": False,
        },
        agents={"defaults": {"workspace": "/tmp/sneaky"}},
    )
    result = apply_config_in_place(live, fresh)

    assert set(result.refused) == {
        "gateway.port",
        "channels.websocket",
        "tools.exec.sandbox",
        "tools.exec.enable",
        "tools.my.allow_set",
        "tools.restrict_to_workspace",
        "agents.defaults.workspace",
    }
    assert (
        live.gateway.port,
        live.channels.websocket["port"],
        live.tools.exec.sandbox,
        live.tools.exec.enable,
        live.tools.my.allow_set,
        live.tools.restrict_to_workspace,
        live.agents.defaults.workspace,
    ) == before, "被拒绝的键必须一个字节都不动"
    assert result.needs_restart is True


def test_model_presets_dict_updates_adds_and_removes_in_place() -> None:
    live = _config(model_presets={"keep": {"model": "m-keep", "provider": "deepseek"}})
    presets_before = live.model_presets
    keep_before = live.model_presets["keep"]

    result = apply_config_in_place(
        live,
        _config(model_presets={
            "keep": {"model": "m-changed", "provider": "deepseek"},
            "added": {"model": "m-added", "provider": "deepseek"},
        }),
    )

    assert live.model_presets is presets_before, "dict 容器本身要原地更新"
    assert live.model_presets["keep"] is keep_before, "已有条目不能替换实例"
    assert live.model_presets["keep"].model == "m-changed"
    assert live.model_presets["added"].model == "m-added"
    assert any(p.startswith("model_presets.keep.") for p in result.applied)

    apply_config_in_place(live, _config(model_presets={"keep": {"model": "m-changed", "provider": "deepseek"}}))
    assert "added" not in live.model_presets, "上游删掉的预设要跟着消失"


def test_custom_provider_extras_are_hot_applied() -> None:
    """自定义服务商走 extra，模型层之外的持有者也要看得见。"""
    live = _config(providers={"custom-token-plan": {"api_key": "sk-old"}})
    providers_before = live.providers
    entry_before = live.providers.__pydantic_extra__["custom-token-plan"]

    result = apply_config_in_place(live, _config(providers={"custom-token-plan": {"api_key": "sk-new"}}))

    assert live.providers is providers_before
    assert any(p.startswith("providers.custom-token-plan.") for p in result.applied)
    assert live.providers.__pydantic_extra__["custom-token-plan"] is entry_before, "条目实例不能被替换"
    assert live.providers.__pydantic_extra__["custom-token-plan"].api_key == "sk-new"


def test_noop_reload_reports_nothing() -> None:
    result = apply_config_in_place(_config(), _config())
    assert result.applied == ()
    assert result.changed is False
    assert result.summary() == "no changes"


def test_is_frozen_covers_registration_time_switches() -> None:
    assert is_frozen("tools.file.enable")
    assert is_frozen("tools.cli_apps.enable")
    assert is_frozen("channels.telegram.token")
    assert is_frozen("agents.defaults.workspace")
    assert is_frozen("tools.exec.max_runtime")
    assert is_frozen("network.proxy")
    assert not is_frozen("agents.defaults.dream.enabled")
    assert not is_frozen("agents.defaults.session_ttl_minutes")
    assert not is_frozen("tools.cli_apps.run_timeout")

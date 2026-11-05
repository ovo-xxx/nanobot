"""Which credential a turn spends: the route decides once, downstream stays as-is."""

import pytest

from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.registry import PROVIDERS
from nanobot.providers.routing import resolve_provider_route


def _route(preset: ModelPresetConfig, providers: dict) -> object:
    config = Config.model_validate({"providers": providers})
    return resolve_provider_route(config, preset=preset)


def _providers(*, base: str = "https://deepseek.test/v1") -> dict:
    return {
        "deepseek": {
            "apiBase": base,
            "apiKey": "sk-personal",
            "credentials": [
                {"id": "work", "name": "公司", "apiKey": "sk-work"},
                {"id": "backup", "apiKey": "sk-backup"},
            ],
        },
    }


def _preset(credential: str = "") -> ModelPresetConfig:
    return ModelPresetConfig(model="deepseek-chat", provider="deepseek", credential=credential)


def test_the_preset_credential_becomes_the_key_the_provider_uses() -> None:
    route = _route(_preset("work"), _providers())

    assert route.provider_config.api_key == "sk-work"
    # nothing else about the connection moved with the key
    assert route.provider_config.api_base == "https://deepseek.test/v1"
    assert route.provider_name == "deepseek"


def test_the_default_key_is_still_reachable_by_name() -> None:
    route = _route(_preset("default"), _providers())

    assert route.provider_config.api_key == "sk-personal"


def test_choosing_between_several_keys_is_not_left_to_order() -> None:
    with pytest.raises(ValueError, match="must name one of"):
        _route(_preset(""), _providers())


def test_a_single_usable_key_needs_no_naming() -> None:
    providers = {"deepseek": {"apiBase": "https://deepseek.test/v1", "credentials": [{"id": "work", "apiKey": "sk-work"}]}}

    route = _route(_preset(""), providers)

    assert route.provider_config.api_key == "sk-work"


def test_a_plain_provider_is_untouched() -> None:
    providers = {"deepseek": {"apiBase": "https://deepseek.test/v1", "apiKey": "sk-only"}}

    for preset in (_preset(""), _preset("default")):
        assert _route(preset, providers).provider_config.api_key == "sk-only"


def test_an_unknown_credential_lists_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match=r"available: default, work \(公司\), backup"):
        _route(_preset("ghost"), _providers())


def test_naming_a_credential_without_a_key_fails_loudly() -> None:
    providers = {
        "deepseek": {
            "apiBase": "https://deepseek.test/v1",
            "apiKey": "sk-personal",
            "credentials": [{"id": "work", "apiKey": "  "}],
        },
    }

    with pytest.raises(ValueError, match="work.*has no api_key"):
        _route(_preset("work"), providers)


def test_oauth_providers_cannot_hold_a_credential_list() -> None:
    oauth = next(spec for spec in PROVIDERS if spec.is_oauth)
    providers = {oauth.name: {"apiKey": "tok", "credentials": [{"id": "second", "apiKey": "tok2"}]}}

    with pytest.raises(ValueError, match="signs in with OAuth"):
        _route(
            ModelPresetConfig(model="any-model", provider=oauth.name, credential="second"),
            providers,
        )

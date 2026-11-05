"""The gateway rebuilds its provider from the config file, so an edit needs no restart.

``watch_config_file`` invalidates the runtime on a change and the next turn loads a
snapshot with ``load_provider_snapshot(config_path)``, which reads the file. These tests
cover the half of that promise the price and credential work depends on: a hand-edit of
the file is what the next call is priced and keyed with.
"""

import pytest

from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.providers.factory import load_provider_snapshot

_PATH = "https://deepseek.test/v1"


def _config(api_key: str, credential_key: str, input_price: float) -> Config:
    return Config.model_validate(
        {
            "providers": {
                "deepseek": {
                    "apiBase": _PATH,
                    "apiKey": api_key,
                    "credentials": [{"id": "work", "apiKey": credential_key}],
                    "pricing": {
                        "unit": "money",
                        "currency": "CNY",
                        "models": {"deepseek-chat": {"offPeak": {"input": input_price, "output": 4}}},
                    },
                }
            },
            "modelPresets": {"primary": {"model": "deepseek-chat", "provider": "deepseek", "credential": "work"}},
        }
    )


def test_a_rewritten_file_changes_the_key_and_the_price_the_next_call_uses(tmp_path) -> None:
    path = tmp_path / "config.json"
    save_config(_config("sk-personal", "sk-work-old", 1), path)

    first = load_provider_snapshot(path, preset_name="primary")
    assert first.provider.api_key == "sk-work-old"
    assert first.provider.credential_id == "work"
    assert first.provider.pricing is not None
    assert first.provider.pricing.models["deepseek-chat"].off_peak.input == 1

    save_config(_config("sk-personal", "sk-work-new", 2), path)

    second = load_provider_snapshot(path, preset_name="primary")
    assert second.provider.api_key == "sk-work-new"
    assert second.provider.pricing is not None
    assert second.provider.pricing.models["deepseek-chat"].off_peak.input == 2
    # the snapshot built from the older file is untouched by the edit
    assert first.provider.api_key == "sk-work-old"


def test_a_preset_naming_a_credential_that_vanished_fails_loudly_rather_than_defaulting(tmp_path) -> None:
    path = tmp_path / "config.json"
    save_config(_config("sk-personal", "sk-work", 1), path)
    assert load_provider_snapshot(path, preset_name="primary").provider.api_key == "sk-work"

    broken = _config("sk-personal", "sk-work", 1).model_dump(by_alias=True)
    broken["providers"]["deepseek"]["credentials"] = []
    save_config(Config.model_validate(broken), path)

    with pytest.raises(ValueError, match="no credential 'work'"):
        load_provider_snapshot(path, preset_name="primary")

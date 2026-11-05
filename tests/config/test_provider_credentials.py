"""Named credentials under one provider: shape, lookup and on-disk form."""

import json

import pytest
from pydantic import ValidationError

from nanobot.config.loader import save_config
from nanobot.config.schema import DEFAULT_CREDENTIAL_ID, Config, ModelPresetConfig


def _provider(**fields):
    return Config.model_validate({"providers": {"deepseek": fields}}).providers.deepseek


def test_the_provider_key_is_the_default_credential() -> None:
    provider = _provider(apiKey="sk-base", credentials=[{"id": "work", "name": "公司", "apiKey": "sk-work"}])

    listed = provider.all_credentials()

    assert [c.id for c in listed] == [DEFAULT_CREDENTIAL_ID, "work"]
    assert listed[0].api_key == "sk-base"
    assert listed[1].api_key == "sk-work"
    assert listed[1].label == "公司"


def test_lookup_is_forgiving_about_shape_and_defaults_to_default() -> None:
    provider = _provider(apiKey="sk-base", credentials=[{"id": "Work-Ratio", "apiKey": "sk-work"}])

    assert provider.credential(None).id == DEFAULT_CREDENTIAL_ID
    assert provider.credential("").id == DEFAULT_CREDENTIAL_ID
    assert provider.credential("work_ratio").id == "work-ratio"
    assert provider.credential("WORK-RATIO").id == "work-ratio"
    assert provider.credential("nope") is None


def test_reserved_and_duplicate_credential_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        _provider(apiKey="sk-base", credentials=[{"id": "default", "apiKey": "sk-x"}])

    with pytest.raises(ValidationError, match="duplicate credential id"):
        _provider(apiKey="sk-base", credentials=[{"id": "a"}, {"id": "a"}])

    with pytest.raises(ValidationError, match="letters, digits and dashes"):
        _provider(apiKey="sk-base", credentials=[{"id": "work key"}])

    with pytest.raises(ValidationError, match="credential id is required"):
        _provider(apiKey="sk-base", credentials=[{"id": "  "}])


def test_a_plain_provider_gains_no_credentials_block_on_disk(tmp_path) -> None:
    """The common config must come back byte-for-byte: no empty list, no copied key."""
    config_path = tmp_path / "config.json"
    config = Config.model_validate({"providers": {"deepseek": {"apiKey": "sk-one"}}})
    save_config(config, config_path)

    stored = json.loads(config_path.read_text("utf-8"))["providers"]["deepseek"]

    assert "credentials" not in stored, "an unused list must not be written into the config"
    assert stored["apiKey"] == "sk-one"


def test_named_credentials_survive_a_save_round_trip(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config = Config.model_validate({
        "providers": {
            "deepseek": {"apiKey": "sk-one", "credentials": [{"id": "work", "name": "公司", "apiKey": "sk-two"}]},
        },
    })
    save_config(config, config_path)

    reloaded = Config.model_validate(json.loads(config_path.read_text("utf-8")))

    assert [c.id for c in reloaded.providers.deepseek.all_credentials()] == [DEFAULT_CREDENTIAL_ID, "work"]
    assert reloaded.providers.deepseek.credential("work").api_key == "sk-two"
    assert json.loads(config_path.read_text("utf-8"))["providers"]["deepseek"]["credentials"][0]["apiKey"] == "sk-two"


def test_has_credential_counts_named_keys_too() -> None:
    assert _provider(apiKey="sk-one").has_credential
    assert not _provider().has_credential
    assert _provider(credentials=[{"id": "work", "apiKey": "sk-two"}]).has_credential
    assert not _provider(apiKey="   ").has_credential, "blank padding is not a credential"


def test_presets_carry_a_credential_id() -> None:
    plain = ModelPresetConfig(model="deepseek-chat")
    picked = ModelPresetConfig(model="deepseek-chat", provider="deepseek", credential="work")

    assert plain.credential == ""
    assert picked.credential == "work"

"""Pricing is arithmetic over the declared rates, done once when the call is recorded."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from nanobot.config.schema import Config
from nanobot.llm_usage.models import LLMCallRecord
from nanobot.llm_usage.types import LLMUsage
from nanobot.pricing.config import ProviderPricing
from nanobot.pricing.cost import cost_for_usage

BEIJING = timezone(timedelta(hours=8))


def _ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    """A wall-clock instant as the service states it; China keeps no daylight saving."""
    return int(datetime(year, month, day, hour, minute, tzinfo=BEIJING).timestamp() * 1000)


def _pricing(payload: dict[str, object]) -> ProviderPricing:
    return ProviderPricing.model_validate(payload)


FLASH = _pricing(
    {
        "unit": "money",
        "currency": "CNY",
        "utcOffsetSeconds": 8 * 3600,
        "band": {"windows": ["09:00-12:00", "14:00-18:00"], "holidays": ["2026-10-01..2026-10-07"]},
        "models": {
            "deepseek-flash": {
                "offPeak": {"input": 1, "cachedInput": 0.02, "output": 4},
                "peak": {"input": 2, "cachedInput": 0.04, "output": 8},
            }
        },
        "source": "https://api.deepseek.com/docs/pricing",
        "asOf": "2026-11-01",
    }
)

# A reseller of the same model: one price all day, and no cache rate declared.
RESELLER = _pricing(
    {
        "unit": "money",
        "currency": "CNY",
        "models": {"deepseek-*": {"offPeak": {"input": 2, "output": 8}}},
    }
)

PLAN = _pricing(
    {
        "unit": "tokens",
        "windows": [{"period": "week", "limit": 50_000_000}],
    }
)


def _usage(*, input_tokens: int = 1000, cache_read: int = 0, output_tokens: int = 500) -> LLMUsage:
    return LLMUsage.reported(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read or None,
    )


def test_a_call_is_priced_by_category() -> None:
    # 1000 fresh input, 90000 cached read, 500 output at the off-peak rates.
    priced = cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 3, 0), model="deepseek-flash", usage=_usage(
        input_tokens=91_000, cache_read=90_000
    ))

    assert priced is not None
    assert priced.input == pytest.approx(0.001)
    assert priced.cache_read == pytest.approx(0.0018)
    assert priced.output == pytest.approx(0.002)
    assert priced.amount == pytest.approx(0.0048)
    assert not priced.peak
    assert priced.currency == "CNY"
    assert not priced.cached_as_input
    assert not priced.estimated


def test_the_same_call_doubles_inside_the_declared_band() -> None:
    usage = _usage(input_tokens=91_000, cache_read=90_000)
    off_peak = cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 3, 0), model="deepseek-flash", usage=usage)
    peak = cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 10, 0), model="deepseek-flash", usage=usage)

    assert off_peak is not None and peak is not None
    assert peak.peak
    assert peak.amount == pytest.approx(off_peak.amount * 2)


def test_the_band_is_read_in_the_services_zone_whatever_zone_the_row_carries() -> None:
    # 02:30 UTC is 10:30 in Beijing, where the service charges its peak rate.
    utc_instant = int(datetime(2026, 11, 4, 2, 30, tzinfo=timezone.utc).timestamp() * 1000)

    priced = cost_for_usage(FLASH, started_at_ms=utc_instant, model="deepseek-flash", usage=_usage())

    assert priced is not None and priced.peak


def test_undeclared_cache_rate_is_the_input_rate_and_says_so() -> None:
    priced = cost_for_usage(
        RESELLER,
        started_at_ms=_ms(2026, 11, 4, 10, 0),
        model="deepseek-flash",
        usage=_usage(input_tokens=2_000, cache_read=1_000),
    )

    assert priced is not None
    # 1000 fresh + 1000 cached at 2元/百万, 500 output at 8元/百万
    assert priced.amount == pytest.approx(0.002 + 0.002 + 0.004)
    assert priced.cached_as_input
    # a service without a band never reports a peak call
    assert not priced.peak


def test_a_model_no_row_covers_prices_as_nothing() -> None:
    assert cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 10, 0), model="gpt-5-mini", usage=_usage()) is None


def test_a_quota_plan_has_no_money_to_report() -> None:
    assert cost_for_usage(PLAN, started_at_ms=_ms(2026, 11, 4, 10, 0), model="any", usage=_usage()) is None


def test_a_request_without_usage_costs_nothing() -> None:
    assert cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 10, 0), model="deepseek-flash", usage=None) is None
    assert (
        cost_for_usage(
            FLASH,
            started_at_ms=_ms(2026, 11, 4, 10, 0),
            model="deepseek-flash",
            usage=LLMUsage.empty_request(),
        )
        is None
    )


def test_an_estimated_usage_is_priced_and_marked_as_not_reported() -> None:
    priced = cost_for_usage(
        FLASH,
        started_at_ms=_ms(2026, 11, 4, 3, 0),
        model="deepseek-flash",
        usage=LLMUsage.estimated(input_tokens=1000, output_tokens=500),
    )

    assert priced is not None
    assert priced.estimated


def test_the_basis_keeps_the_rates_and_their_provenance() -> None:
    priced = cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 10, 0), model="deepseek-flash", usage=_usage(
        input_tokens=91_000, cache_read=90_000
    ))

    assert priced is not None
    basis = json.loads(priced.basis())
    assert basis["row"] == "deepseek-flash"
    assert basis["peak"] is True
    # the peak tier the amount was actually built from, per one million tokens
    assert basis["rates"] == [2.0, 0.04, 8.0]
    assert basis["breakdown"]["input"] == pytest.approx(0.002)
    assert basis["breakdown"]["cacheRead"] == pytest.approx(0.0036)
    assert basis["breakdown"]["output"] == pytest.approx(0.004)
    assert basis["cachedAsInput"] is False
    assert basis["source"] == "https://api.deepseek.com/docs/pricing"
    assert basis["asOf"] == "2026-11-01"
    # the amount and the currency live in their own columns, not repeated in here
    assert "amount" not in basis
    assert "currency" not in basis


def test_the_block_round_trips_through_the_provider_config() -> None:
    config = Config.model_validate(
        {"providers": {"deepseek": {"apiBase": "https://x.test/v1", "pricing": FLASH}}}
    )
    provider = config.providers.deepseek

    assert provider is not None
    assert provider.pricing is not None
    assert provider.pricing.models["deepseek-flash"].off_peak.cached_input == 0.02
    # the config file keeps the block in the camelCase the rest of the file uses
    dumped = provider.model_dump(by_alias=True, exclude_none=True)
    assert dumped["pricing"]["utcOffsetSeconds"] == 28800
    assert dumped["pricing"]["models"]["deepseek-flash"]["offPeak"]["cachedInput"] == 0.02


def test_a_record_carries_its_price_into_the_store(tmp_path) -> None:
    from nanobot.llm_usage.store import LLMUsageStore

    priced = cost_for_usage(FLASH, started_at_ms=_ms(2026, 11, 4, 10, 0), model="deepseek-flash", usage=_usage())
    assert priced is not None
    store = LLMUsageStore(tmp_path / "llm_usage.sqlite3")
    store.record(
        LLMCallRecord(
            started_at_ms=_ms(2026, 11, 4, 10, 0),
            duration_ms=400,
            provider="deepseek",
            model="deepseek-flash",
            source="user",
            stream=True,
            finish_reason="stop",
            usage=_usage(),
            credential="work",
            amount=priced.amount,
            currency=priced.currency,
            price_basis=priced.basis(),
        )
    )

    row = store.recent_calls(limit=1)[0]
    assert row["amount"] == pytest.approx(0.006)
    assert row["currency"] == "CNY"
    assert json.loads(row["price_basis"])["peak"] is True

    # a call the price table does not cover stores no amount rather than a zero
    store.record(
        LLMCallRecord(
            started_at_ms=_ms(2026, 11, 4, 10, 5),
            duration_ms=400,
            provider="deepseek",
            model="gpt-5-mini",
            source="user",
            stream=True,
            finish_reason="stop",
            usage=_usage(),
        )
    )
    rows = store.recent_calls(limit=2)
    assert [row["amount"] for row in rows] == [None, pytest.approx(0.006)]

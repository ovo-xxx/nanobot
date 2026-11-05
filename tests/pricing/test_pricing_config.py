"""The shape of a price block: what a service is allowed to declare, and how it reads."""

from __future__ import annotations

from datetime import date

import pytest

from nanobot.config.schema import Config
from nanobot.pricing.config import LocalTime, PeakBand, ProviderPricing


def _at(year: int, month: int, day: int, hour: int, minute: int) -> LocalTime:
    return LocalTime(day=date(year, month, day), minutes_of_day=hour * 60 + minute)


def _deepseek_like(**overrides: object) -> ProviderPricing:
    payload: dict[str, object] = {
        "unit": "money",
        "currency": "CNY",
        "utcOffsetSeconds": 8 * 3600,
        "band": {
            "windows": ["09:00-12:00", "14:00-18:00"],
            "holidays": ["2026-10-01..2026-10-07"],
        },
        "models": {
            "deepseek-flash": {
                "offPeak": {"input": 1, "cachedInput": 0.02, "output": 4},
                "peak": {"input": 2, "cachedInput": 0.04, "output": 8},
            }
        },
        "source": "https://api.deepseek.com/docs/pricing",
        "asOf": "2026-11-01",
    }
    payload.update(overrides)
    return ProviderPricing.model_validate(payload)


def test_camel_case_and_snake_case_both_read() -> None:
    pricing = _deepseek_like()

    assert pricing.utc_offset_seconds == 28800
    assert pricing.models["deepseek-flash"].peak is not None
    assert pricing.as_of == "2026-11-01"


def test_a_provider_without_a_price_block_writes_no_block_back() -> None:
    config = Config.model_validate({"providers": {"deepseek": {"apiBase": "https://x.test/v1"}}})

    assert config.providers.deepseek is not None
    assert config.providers.deepseek.pricing is None
    assert "pricing" not in config.providers.deepseek.model_dump()


def test_band_reads_windows_as_half_open_and_days_as_a_veto() -> None:
    band = _deepseek_like().band
    assert band is not None

    assert band.is_peak(_at(2026, 11, 4, 9, 0))  # Wednesday, opens the slot
    assert band.is_peak(_at(2026, 11, 4, 11, 59))
    assert not band.is_peak(_at(2026, 11, 4, 12, 0))  # excluded at the close
    assert not band.is_peak(_at(2026, 11, 4, 13, 0))  # lunch
    assert not band.is_peak(_at(2026, 11, 7, 10, 0))  # Saturday is never peak
    assert not band.is_peak(_at(2026, 10, 5, 10, 0))  # a weekday inside the holiday


def test_a_wrapping_window_belongs_to_the_day_it_starts_on() -> None:
    band = PeakBand(windows=["22:00-02:00"], days=[1, 2, 3, 4, 5, 6, 7])

    assert band.is_peak(_at(2026, 11, 6, 23, 30))  # Friday night
    assert band.is_peak(_at(2026, 11, 7, 1, 0))  # still Friday's window, Saturday clock
    assert not band.is_peak(_at(2026, 11, 7, 9, 0))


def test_a_service_without_a_band_prices_every_hour_the_same() -> None:
    pricing = _deepseek_like(band=None)

    assert pricing.band is None
    assert pricing.has_rates


def test_prefix_rows_cover_a_family_and_the_longest_wins() -> None:
    pricing = _deepseek_like(
        models={
            "deepseek-*": {"offPeak": {"input": 1, "output": 4}},
            "deepseek-chat": {"offPeak": {"input": 3, "output": 9}},
        }
    )

    assert pricing.pricing_for("deepseek-chat") is not None
    exact, row = pricing.pricing_for("deepseek-chat")
    assert exact == "deepseek-chat"
    assert row.off_peak.input == 3

    prefix, family = pricing.pricing_for("deepseek-reasoner")
    assert prefix == "deepseek-*"
    assert family.off_peak.input == 1

    assert pricing.pricing_for("gpt-5-mini") is None


def test_a_model_may_declare_only_an_off_peak_tier() -> None:
    pricing = _deepseek_like(
        models={"deepseek-flash": {"offPeak": {"input": 1, "cachedInput": 0.02, "output": 4}}}
    )

    row = pricing.pricing_for("deepseek-flash")
    assert row is not None
    _, prices = row
    assert prices.rates(peak=True) is prices.off_peak


def test_money_pricing_must_name_a_currency() -> None:
    with pytest.raises(ValueError, match="currency"):
        _deepseek_like(currency="")


def test_a_quota_has_no_time_of_day_tier() -> None:
    with pytest.raises(ValueError, match="no time-of-day tier"):
        ProviderPricing.model_validate(
            {
                "unit": "tokens",
                "band": {"windows": ["09:00-12:00"]},
                "windows": [{"period": "month", "limit": 1000000}],
            }
        )


def test_window_periods_may_not_repeat() -> None:
    with pytest.raises(ValueError, match="window period twice"):
        _deepseek_like(
            windows=[
                {"period": "week", "limit": 10},
                {"period": "week", "limit": 20},
            ]
        )


def test_a_monthly_window_may_anchor_the_billing_day() -> None:
    pricing = _deepseek_like(windows=[{"period": "month", "limit": 1200, "anchorDay": 15}])

    assert pricing.windows[0].anchor_day == 15


def test_anchor_day_only_makes_sense_for_a_month() -> None:
    with pytest.raises(ValueError, match="anchor_day only applies"):
        _deepseek_like(windows=[{"period": "week", "limit": 1200, "anchorDay": 15}])


def test_malformed_declarations_are_rejected_on_load() -> None:
    for payload, message in (
        ({"band": {"windows": ["9-12"]}}, "09:00-12:00"),
        ({"band": {"holidays": ["2026-10-07..2026-10-01"]}}, "starts after it ends"),
        ({"band": {"holidays": ["National Day"]}}, "not a 2026-10-01 date"),
        ({"band": {"days": [8]}}, "peak day 8"),
        ({"asOf": "last Tuesday"}, "not a 2026-10-01 date"),
        ({"models": {"deepseek-flash": {"offPeak": {"input": 1, "output": -4}}}}, "greater than or equal"),
    ):
        with pytest.raises(ValueError, match=message):
            _deepseek_like(**payload)

"""Pricing is arithmetic over declared rates: per call, per quota window, per service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nanobot.config.schema import Config
from nanobot.llm_usage.models import LLMCallRecord
from nanobot.llm_usage.types import LLMUsage
from nanobot.pricing import cost
from nanobot.pricing.config import ProviderPricing

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
        "windows": [
            {"period": "hour", "limit": 1_000_000},
            {"period": "day", "limit": 5_000_000},
            {"period": "week", "limit": 50_000_000},
            {"period": "month", "limit": 200_000_000, "anchorDay": 15},
        ],
    }
)


def _usage(*, input_tokens: int = 1000, cache_read: int = 0, output_tokens: int = 500) -> LLMUsage:
    return LLMUsage.reported(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read or None,
    )


def _call(
    started_at_ms: int,
    *,
    provider: str = "deepseek",
    model: str = "deepseek-flash",
    usage: LLMUsage | None = None,
) -> LLMCallRecord:
    return LLMCallRecord(
        started_at_ms=started_at_ms,
        duration_ms=400,
        provider=provider,
        model=model,
        source="user",
        stream=True,
        finish_reason="stop",
        usage=usage if usage is not None else _usage(),
    )


def test_a_call_is_priced_by_category() -> None:
    # 1000 fresh input, 90000 cached read, 500 output at the off-peak rates.
    usage = _usage(input_tokens=91_000, cache_read=90_000, output_tokens=500)

    priced = cost.cost_for_call(
        FLASH,
        started_at_ms=_ms(2026, 11, 4, 3, 0),
        model="deepseek-flash",
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens or 0,
    )

    assert priced is not None
    assert priced.input == pytest.approx(0.001)
    assert priced.cache_read == pytest.approx(0.0018)
    assert priced.output == pytest.approx(0.002)
    assert priced.amount == pytest.approx(0.0048)
    assert not priced.peak
    assert priced.currency == "CNY"
    assert not priced.cached_as_input
    # the tooltip needs the basis, not just the total
    assert priced.row == "deepseek-flash"
    assert priced.source == "https://api.deepseek.com/docs/pricing"
    assert priced.as_of == "2026-11-01"


def test_the_same_call_doubles_inside_the_declared_band() -> None:
    usage = _usage(input_tokens=91_000, cache_read=90_000, output_tokens=500)
    off_peak = cost.cost_for_call(
        FLASH,
        started_at_ms=_ms(2026, 11, 4, 3, 0),
        model="deepseek-flash",
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens or 0,
    )
    peak = cost.cost_for_call(
        FLASH,
        started_at_ms=_ms(2026, 11, 4, 10, 0),
        model="deepseek-flash",
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens or 0,
    )

    assert off_peak is not None and peak is not None
    assert peak.peak
    assert peak.amount == pytest.approx(off_peak.amount * 2)


def test_the_band_is_read_in_the_services_zone_whatever_zone_the_row_carries() -> None:
    # 02:30 UTC is 10:30 in Beijing, where the service charges its peak rate.
    utc_instant = int(datetime(2026, 11, 4, 2, 30, tzinfo=timezone.utc).timestamp() * 1000)
    usage = _usage()

    priced = cost.cost_for_call(
        FLASH,
        started_at_ms=utc_instant,
        model="deepseek-flash",
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )

    assert priced is not None and priced.peak


def test_undeclared_cache_rate_is_the_input_rate_and_says_so() -> None:
    usage = _usage(input_tokens=2_000, cache_read=1_000, output_tokens=500)

    priced = cost.cost_for_call(
        RESELLER,
        started_at_ms=_ms(2026, 11, 4, 10, 0),
        model="deepseek-flash",
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens or 0,
    )

    assert priced is not None
    # 1000 fresh + 1000 cached at 2元/百万, 500 output at 8元/百万
    assert priced.amount == pytest.approx(0.002 + 0.002 + 0.004)
    assert priced.cached_as_input
    # a service without a band never reports a peak call
    assert not priced.peak


def test_a_model_no_row_covers_prices_as_nothing() -> None:
    usage = _usage()

    assert (
        cost.cost_for_call(
            FLASH,
            started_at_ms=_ms(2026, 11, 4, 10, 0),
            model="gpt-5-mini",
            total_tokens=usage.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        is None
    )


def test_a_quota_plan_has_no_money_to_show() -> None:
    usage = _usage()

    assert (
        cost.cost_for_call(
            PLAN,
            started_at_ms=_ms(2026, 11, 4, 10, 0),
            model="any",
            total_tokens=usage.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        is None
    )


def test_windows_count_the_current_period_only() -> None:
    def tokens(total: int) -> LLMUsage:
        return LLMUsage.reported(input_tokens=total, output_tokens=0)

    rows = [
        _call(_ms(2026, 11, 4, 10, 59), usage=tokens(100)),  # earlier today, previous hour
        _call(_ms(2026, 11, 4, 11, 30), usage=tokens(200)),  # inside the hour we are in
        _call(_ms(2026, 11, 2, 9, 0), usage=tokens(400)),  # Monday, still this week
        _call(_ms(2026, 10, 30, 9, 0), usage=tokens(800)),  # last week, inside the month cycle
        _call(_ms(2026, 11, 10, 9, 0), usage=tokens(1600)),  # next week, inside the month cycle
        _call(_ms(2026, 11, 16, 9, 0), usage=tokens(3200)),  # after the cycle rolls over
    ]

    windows = cost.window_usage(PLAN, rows, {}, now_ms=_ms(2026, 11, 4, 11, 59))

    by_period = {window.period: window for window in windows}
    assert by_period["hour"].used == 200
    assert by_period["day"].used == 300
    assert by_period["week"].used == 700
    # the anchored cycle runs on the 15th, so 30 Oct - 14 Nov is one month for this plan
    assert by_period["month"].used == 3100
    assert by_period["week"].ratio == pytest.approx(700 / 50_000_000)


def test_a_cycle_anchored_on_the_31st_rolls_with_the_month() -> None:
    def tokens(total: int) -> LLMUsage:
        return LLMUsage.reported(input_tokens=total, output_tokens=0)

    anchored = _pricing(
        {"unit": "tokens", "windows": [{"period": "month", "limit": 100_000, "anchorDay": 31}]}
    )
    rows = [
        _call(_ms(2026, 10, 31, 9, 0), usage=tokens(10)),  # the cycle begins here
        _call(_ms(2026, 11, 29, 23, 0), usage=tokens(20)),  # last instant of the cycle
        _call(_ms(2026, 11, 30, 9, 0), usage=tokens(40)),  # a shorter month starts the next one
    ]

    inside = cost.window_usage(anchored, rows, {}, now_ms=_ms(2026, 11, 15, 12, 0))
    rolling = cost.window_usage(anchored, rows, {}, now_ms=_ms(2026, 11, 30, 12, 0))

    assert inside[0].used == 30
    assert rolling[0].used == 40


def test_a_money_service_can_show_spend_per_window() -> None:
    off_peak = _call(_ms(2026, 11, 4, 3, 0), usage=_usage(input_tokens=91_000, cache_read=90_000))
    peak = _call(_ms(2026, 11, 4, 10, 0), usage=_usage(input_tokens=91_000, cache_read=90_000))
    money = [
        _pricing(
            {
                "unit": "money",
                "currency": "CNY",
                "utcOffsetSeconds": 8 * 3600,
                "models": {"deepseek-flash": {"offPeak": {"input": 1, "cachedInput": 0.02, "output": 4}}},
                "windows": [{"period": "day", "limit": 20}],
            }
        )
    ]

    summary = cost.summarize_provider("deepseek", money, [off_peak, peak], now_ms=_ms(2026, 11, 4, 12, 0))

    assert summary is not None
    assert summary.amount == pytest.approx(0.0048 * 2)
    assert summary.counted_calls == 2
    assert summary.windows[0].used == pytest.approx(0.0048 * 2)
    assert summary.windows[0].limit == 20


def test_a_revision_keeps_the_older_days_at_their_own_price() -> None:
    """The reason amounts are not written to the table: the history is the price list."""
    history = [
        _pricing(
            {
                "unit": "money",
                "currency": "CNY",
                "utcOffsetSeconds": 8 * 3600,
                "models": {"deepseek-flash": {"offPeak": {"input": 1, "output": 4}}},
            }
        ),
        _pricing(
            {
                "unit": "money",
                "currency": "CNY",
                "effectiveFrom": "2026-11-01",
                "utcOffsetSeconds": 8 * 3600,
                "models": {"deepseek-flash": {"offPeak": {"input": 2, "output": 8}}},
            }
        ),
    ]
    before = _call(_ms(2026, 10, 20, 10, 0), usage=_usage(input_tokens=1_000))
    after = _call(_ms(2026, 11, 20, 10, 0), usage=_usage(input_tokens=1_000))

    summary = cost.summarize_provider("deepseek", history, [before, after], now_ms=_ms(2026, 11, 30, 0, 0))

    assert summary is not None
    # 1000 fresh input + 500 output, at 1/4 then at the doubled 2/8
    assert summary.amount == pytest.approx(0.001 + 0.002 + 0.002 + 0.004)
    single = cost.cost_for_call(
        history[0],
        started_at_ms=before.started_at_ms,
        model="deepseek-flash",
        total_tokens=1_500,
        input_tokens=1_000,
        output_tokens=500,
    )
    assert single is not None
    assert single.amount == pytest.approx(0.003)


def test_summaries_separate_priced_calls_from_calls_no_row_covers() -> None:
    rows = [
        _call(_ms(2026, 11, 4, 10, 0)),
        _call(_ms(2026, 11, 4, 10, 5), model="gpt-5-mini"),
        _call(_ms(2026, 11, 4, 10, 6), usage=LLMUsage.empty_request()),
        _call(_ms(2026, 11, 4, 10, 7), provider="plan", usage=_usage(input_tokens=5_000)),
    ]
    blocks = {"deepseek": [FLASH], "plan": [PLAN]}

    summaries = cost.summarize_by_provider(
        rows, pricing_of=lambda name: blocks.get(name, []), now_ms=_ms(2026, 11, 4, 12, 0)
    )

    assert [summary.provider for summary in summaries] == ["deepseek", "plan"]
    deepseek, plan = summaries
    assert deepseek.counted_calls == 1
    assert deepseek.peak_calls == 1
    # the call no price row covers is counted as unpriced, not priced at zero
    assert deepseek.unpriced_calls == 1
    assert plan.unit == "tokens"
    # a quota service counts every call with tokens, and has no money to report
    assert plan.counted_calls == 1
    assert plan.amount == 0.0


def test_a_service_without_a_price_block_is_left_out_rather_than_priced_at_zero() -> None:
    summaries = cost.summarize_by_provider(
        [_call(_ms(2026, 11, 4, 10, 0))],
        pricing_of=lambda provider: [],
        now_ms=_ms(2026, 11, 4, 12, 0),
    )

    assert summaries == []


def test_the_blocks_round_trip_through_the_provider_config() -> None:
    config = Config.model_validate(
        {"providers": {"deepseek": {"apiBase": "https://x.test/v1", "pricing": [FLASH]}}}
    )
    provider = config.providers.deepseek

    assert provider is not None
    assert provider.pricing[0].models["deepseek-flash"].off_peak.cached_input == 0.02
    # the config file keeps the history in the camelCase the rest of the file uses
    dumped = provider.model_dump(by_alias=True, exclude_none=True)
    assert dumped["pricing"][0]["utcOffsetSeconds"] == 28800
    assert dumped["pricing"][0]["models"]["deepseek-flash"]["offPeak"]["cachedInput"] == 0.02

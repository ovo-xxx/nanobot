"""Turn a service's token counts into the number that service would show you.

Nothing here is stored: rows keep token counts, and the money or the quota occupancy
is computed from the service's price history each time it is read. A call is priced
against the list that was in force on its own day, so a revision adds a dated block and
leaves the older days' totals where they were - which is the whole reason the amounts
are not written into the table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Literal

from nanobot.llm_usage.models import LLMCallRecord
from nanobot.pricing.config import LocalTime, ProviderPricing, block_on

PER_MILLION = 1_000_000.0

Period = Literal["hour", "day", "week", "month"]


@dataclass(frozen=True, slots=True)
class CallCost:
    """The price of one call, keeping the rates behind it for the tooltip."""

    row: str
    peak: bool
    currency: str
    amount: float
    input: float
    cache_read: float
    output: float
    cached_as_input: bool
    estimated: bool
    source: str
    as_of: str


@dataclass(frozen=True, slots=True)
class WindowUsage:
    period: Period
    limit: float
    used: float

    @property
    def ratio(self) -> float:
        return self.used / self.limit if self.limit > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ProviderSummary:
    provider: str
    unit: str
    currency: str
    amount: float
    counted_calls: int
    peak_calls: int
    unpriced_calls: int
    windows: tuple[WindowUsage, ...] = ()


def local_time(started_at_ms: int, utc_offset_seconds: int) -> datetime:
    return datetime.fromtimestamp(started_at_ms / 1000, tz=timezone.utc).astimezone(
        timezone(timedelta(seconds=utc_offset_seconds))
    )


def _at(started_at_ms: int, pricing: ProviderPricing) -> LocalTime:
    moment = local_time(started_at_ms, pricing.utc_offset_seconds)
    return LocalTime(day=moment.date(), minutes_of_day=moment.hour * 60 + moment.minute)


def _day_of(started_at_ms: int, pricing: ProviderPricing) -> date:
    return local_time(started_at_ms, pricing.utc_offset_seconds).date()


def cost_for_call(
    pricing: ProviderPricing | None,
    *,
    started_at_ms: int,
    model: str,
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    estimated: bool = False,
) -> CallCost | None:
    """Price one call, or return nothing when the service declares no rate for it.

    ``input_tokens`` already contains the cache categories, so the fresh input is what
    is left once they are taken out. Cache writes are billed as fresh input, and cache
    reads fall back to the input rate when the service declares no cache rate - both
    are reported back on the cost so the tooltip can say which was used.
    """
    if pricing is None or pricing.unit != "money" or total_tokens <= 0:
        return None
    matched = pricing.pricing_for(model)
    if matched is None:
        return None
    row, prices = matched
    peak = pricing.band is not None and pricing.band.is_peak(_at(started_at_ms, pricing))
    rates = prices.rates(peak=peak)
    declared_cache = rates.cached_input
    cached_as_input = declared_cache is None
    cached_rate = rates.input if declared_cache is None else declared_cache
    fresh_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    amount_input = fresh_input / PER_MILLION * rates.input
    amount_cache = cache_read_tokens / PER_MILLION * cached_rate
    amount_output = output_tokens / PER_MILLION * rates.output
    return CallCost(
        row=row,
        peak=peak,
        currency=pricing.currency,
        amount=amount_input + amount_cache + amount_output,
        input=amount_input,
        cache_read=amount_cache,
        output=amount_output,
        cached_as_input=cached_as_input,
        estimated=estimated,
        source=pricing.source,
        as_of=pricing.as_of,
    )


def _cost_of(call: LLMCallRecord, pricing: ProviderPricing) -> CallCost | None:
    usage = call.usage
    if usage is None:
        return None
    return cost_for_call(
        pricing,
        started_at_ms=call.started_at_ms,
        model=call.model,
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens or 0,
        cache_write_tokens=usage.cache_write_tokens or 0,
        estimated=usage.source != "reported",
    )


def _midnight(day: date, zone: tzinfo | None) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=zone)


def _period_start(now: datetime, period: Period, anchor_day: int | None) -> datetime:
    day = now.date()
    if period == "hour":
        return now.replace(minute=0, second=0, microsecond=0)
    if period == "day":
        return _midnight(day, now.tzinfo)
    if period == "week":
        return _midnight(day - timedelta(days=day.isoweekday() - 1), now.tzinfo)
    if anchor_day is None:
        return _midnight(day.replace(day=1), now.tzinfo)
    # A cycle anchored past the end of a short month rolls to its last day, and the
    # comparison has to be made against that rolled day or the whole short month
    # would read as the previous cycle.
    rolls_on = min(anchor_day, _days_in_month(day))
    month_start = day if day.day >= rolls_on else _previous_month(day)
    return _midnight(month_start.replace(day=min(anchor_day, _days_in_month(month_start))), now.tzinfo)


def _period_end(start: datetime, period: Period, anchor_day: int | None) -> datetime:
    """The instant the next period begins, so a window stays half-open."""
    if period == "hour":
        return start + timedelta(hours=1)
    if period == "day":
        return start + timedelta(days=1)
    if period == "week":
        return start + timedelta(days=7)
    if anchor_day is None:
        return _midnight(_next_month(start.date()), start.tzinfo)
    following = _next_month(start.date())
    return _midnight(following.replace(day=min(anchor_day, _days_in_month(following))), start.tzinfo)


def _next_month(day: date) -> date:
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def _days_in_month(day: date) -> int:
    return (_next_month(day) - day.replace(day=1)).days


def _previous_month(day: date) -> date:
    return day.replace(day=1) - timedelta(days=1)


def _counts_toward_quota(call: LLMCallRecord) -> bool:
    return call.usage is not None and call.usage.total_tokens > 0


def window_usage(
    pricing: ProviderPricing,
    calls: list[LLMCallRecord],
    amounts: dict[int, float],
    *,
    now_ms: int,
) -> tuple[WindowUsage, ...]:
    """Occupancy of each quota window the service declares, counted in its own unit."""
    if not pricing.windows:
        return ()
    now = local_time(now_ms, pricing.utc_offset_seconds)
    rows = [call for call in calls if _counts_toward_quota(call)]
    usage: list[WindowUsage] = []
    for window in pricing.windows:
        start = _period_start(now, window.period, window.anchor_day)
        end = _period_end(start, window.period, window.anchor_day)
        inside = [
            call
            for call in rows
            if start <= local_time(call.started_at_ms, pricing.utc_offset_seconds) < end
        ]
        if pricing.unit == "money":
            used = sum(amounts.get(call.started_at_ms, 0.0) for call in inside)
        else:
            used = float(sum(call.usage.total_tokens for call in inside if call.usage is not None))
        usage.append(WindowUsage(period=window.period, limit=window.limit, used=used))
    return tuple(usage)


def summarize_provider(
    provider: str,
    blocks: Sequence[ProviderPricing],
    calls: Iterable[LLMCallRecord],
    *,
    now_ms: int,
) -> ProviderSummary | None:
    """What one service's rows add up to, or nothing if it declares no price block.

    Each call is priced against the list in force on its own day, so a revision prices
    the days after it and leaves the older days at the money they actually ran under.
    """
    if not blocks:
        return None
    rows = [call for call in calls if _counts_toward_quota(call)]
    current = max(blocks, key=lambda block: block.starts_on)
    amount = 0.0
    peak_calls = 0
    unpriced_calls = 0
    amounts: dict[int, float] = {}
    for call in rows:
        if current.unit != "money":
            continue
        pricing = block_on(_day_of(call.started_at_ms, current), blocks)
        if pricing is None or pricing.unit != "money":
            unpriced_calls += 1
            continue
        cost = _cost_of(call, pricing)
        if cost is None:
            unpriced_calls += 1
            continue
        amount += cost.amount
        peak_calls += 1 if cost.peak else 0
        amounts[call.started_at_ms] = amounts.get(call.started_at_ms, 0.0) + cost.amount
    counted = len(rows) if current.unit != "money" else len(rows) - unpriced_calls
    return ProviderSummary(
        provider=provider,
        unit=current.unit,
        currency=current.currency,
        amount=amount,
        counted_calls=counted,
        peak_calls=peak_calls,
        unpriced_calls=unpriced_calls,
        windows=window_usage(current, rows, amounts, now_ms=now_ms),
    )


def summarize_by_provider(
    calls: Iterable[LLMCallRecord],
    *,
    pricing_of: Callable[[str], Sequence[ProviderPricing]],
    now_ms: int,
) -> list[ProviderSummary]:
    """Per-service summaries, one per provider that declares a price block."""
    grouped: dict[str, list[LLMCallRecord]] = {}
    for call in calls:
        grouped.setdefault(call.provider, []).append(call)
    return [
        summary
        for provider, rows in grouped.items()
        for summary in [summarize_provider(provider, pricing_of(provider), rows, now_ms=now_ms)]
        if summary is not None
    ]


__all__ = [
    "CallCost",
    "ProviderSummary",
    "WindowUsage",
    "cost_for_call",
    "local_time",
    "summarize_by_provider",
    "summarize_provider",
    "window_usage",
]

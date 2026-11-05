"""Price one call against the service's declared rate table.

The arithmetic happens when the call is recorded, and the outcome is stored with the
row: the amount, the currency it is stated in, and the rates that produced it. Money
therefore never moves after the fact - a service revising its prices changes what the
next call costs and leaves last month's total alone - and a report can still show the
basis of a number whose price list no longer exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from nanobot.llm_usage.types import LLMUsage
from nanobot.pricing.config import LocalTime, ProviderPricing

PER_MILLION = 1_000_000.0


@dataclass(frozen=True, slots=True)
class CallCost:
    """The price of one call, with the rates behind it kept for the tooltip."""

    row: str
    peak: bool
    currency: str
    amount: float
    input: float
    cache_read: float
    output: float
    rates: tuple[float, float, float]
    cached_as_input: bool
    estimated: bool
    source: str
    as_of: str

    def basis(self) -> str:
        """The rates behind this cost, stored beside the amount for later display.

        Written as one JSON object and read back by the panel, so every field keeps the
        spelling it has here instead of a per-field rename list drifting out of date.
        """
        return json.dumps(
            {
                "row": self.row,
                "peak": self.peak,
                "rates": list(self.rates),
                "breakdown": {
                    "input": self.input,
                    "cacheRead": self.cache_read,
                    "output": self.output,
                },
                "cachedAsInput": self.cached_as_input,
                "estimated": self.estimated,
                "source": self.source,
                "asOf": self.as_of,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def local_time(started_at_ms: int, utc_offset_seconds: int) -> datetime:
    return datetime.fromtimestamp(started_at_ms / 1000, tz=timezone.utc).astimezone(
        timezone(timedelta(seconds=utc_offset_seconds))
    )


def _at(started_at_ms: int, pricing: ProviderPricing) -> LocalTime:
    moment = local_time(started_at_ms, pricing.utc_offset_seconds)
    return LocalTime(day=moment.date(), minutes_of_day=moment.hour * 60 + moment.minute)


def cost_for_usage(
    pricing: ProviderPricing | None,
    *,
    started_at_ms: int,
    model: str,
    usage: LLMUsage | None,
) -> CallCost | None:
    """Price one call, or return nothing when the service declares no rate for it.

    ``usage.input_tokens`` already contains the cache categories, so the fresh input is
    what is left once they are taken out. Cache writes are billed as fresh input, and
    cache reads fall back to the input rate when the service declares no cache rate -
    both are reported back so the tooltip can say which was used.
    """
    if pricing is None or pricing.unit != "money" or usage is None or usage.total_tokens <= 0:
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
    fresh_input = max(0, usage.input_tokens - (usage.cache_read_tokens or 0) - (usage.cache_write_tokens or 0))
    amount_input = fresh_input / PER_MILLION * rates.input
    amount_cache = (usage.cache_read_tokens or 0) / PER_MILLION * cached_rate
    amount_output = usage.output_tokens / PER_MILLION * rates.output
    return CallCost(
        row=row,
        peak=peak,
        currency=pricing.currency,
        amount=amount_input + amount_cache + amount_output,
        input=amount_input,
        cache_read=amount_cache,
        output=amount_output,
        rates=(rates.input, cached_rate, rates.output),
        cached_as_input=cached_as_input,
        estimated=usage.source != "reported",
        source=pricing.source,
        as_of=pricing.as_of,
    )


__all__ = ["CallCost", "cost_for_usage", "local_time"]

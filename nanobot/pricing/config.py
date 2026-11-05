"""Per-service price declaration.

The service decides what a number means: the official DeepSeek endpoint prices by
the token and charges more inside its peak window, a reseller of the same model may
keep one price all day, and a subscription plan sells a quota, not a price. So all of
it hangs on ``providers.<name>.pricing`` and nothing is inferred from the model name.

The block is the price list in force from now on. A call is priced when it is
recorded, and the money it cost is stored with it, so a later revision of this block
leaves the older rows at the price they actually ran at.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal, NamedTuple

from pydantic import Field, field_validator, model_validator

from nanobot.config_base import Base

PriceUnit = Literal["money", "tokens"]


class LocalTime(NamedTuple):
    """A local calendar day plus the minutes elapsed in it, which is all a band reads."""

    day: date
    minutes_of_day: int

    @property
    def weekday(self) -> int:
        return self.day.isoweekday()


class ModelRates(Base):
    """One price tier, in the provider's own currency per one million tokens."""

    input: float = Field(ge=0)
    cached_input: float | None = Field(default=None, ge=0)
    output: float = Field(ge=0)


class ModelPricing(Base):
    """Prices for one model of one service.

    ``peak`` is only read when the service declares a band; a service without a band
    prices every call from ``off_peak``.
    """

    off_peak: ModelRates
    peak: ModelRates | None = None

    def rates(self, *, peak: bool) -> ModelRates:
        return self.peak if (peak and self.peak is not None) else self.off_peak


class PeakBand(Base):
    """When the service charges its peak tier, as the service states it.

    Times are wall-clock at ``ProviderPricing.utc_offset_seconds``, intervals are
    half-open (``09:00-12:00`` includes 09:00 and excludes 12:00), and a window that
    wraps past midnight belongs to the day it starts on. ``days`` and ``holidays``
    win over ``windows``: a day outside the list, or a holiday, is never peak.
    """

    windows: list[str] = Field(default_factory=list)
    days: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    holidays: list[str] = Field(default_factory=list)

    @field_validator("windows")
    @classmethod
    def _validate_windows(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        for window in cleaned:
            if _WINDOW.match(window) is None:
                raise ValueError(f"peak window {window!r} must look like 09:00-12:00")
        return cleaned

    @field_validator("days")
    @classmethod
    def _validate_days(cls, value: list[int]) -> list[int]:
        for day in value:
            if not 1 <= day <= 7:
                raise ValueError(f"peak day {day} must be 1 (Monday) to 7 (Sunday)")
        return sorted(set(value))

    @field_validator("holidays")
    @classmethod
    def _validate_holidays(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        for entry in cleaned:
            _holiday_range(entry)
        return cleaned

    def is_peak(self, at: LocalTime) -> bool:
        if at.weekday not in self.days:
            return False
        if _in_any_range(at.day, self.holidays):
            return False
        return any(_in_window(window, at.minutes_of_day) for window in self.windows)


class PricingWindow(Base):
    """A quota period the service counts, in the unit the service sells in.

    ``anchor_day`` is the day of the month the service rolls its cycle on - the
    subscription date, not the calendar month. A cycle anchored on the 31st rolls to
    the last day of a shorter month, which is what those services bill.
    """

    period: Literal["hour", "day", "week", "month"]
    limit: float = Field(gt=0)
    anchor_day: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _validate_anchor(self) -> "PricingWindow":
        if self.anchor_day is not None and self.period != "month":
            raise ValueError("anchor_day only applies to a monthly window")
        return self


class ProviderPricing(Base):
    """The price list one service currently sells under."""

    unit: PriceUnit = "money"
    currency: str = ""
    utc_offset_seconds: int = Field(default=0, ge=-43200, le=50400)
    band: PeakBand | None = None
    models: dict[str, ModelPricing] = Field(default_factory=dict)
    windows: list[PricingWindow] = Field(default_factory=list)
    source: str = ""
    as_of: str = ""

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned and not _CURRENCY.match(cleaned):
            raise ValueError(f"currency {value!r} must be a three-letter code like CNY")
        return cleaned

    @field_validator("as_of")
    @classmethod
    def _validate_as_of(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned:
            _parse_day(cleaned)
        return cleaned

    @model_validator(mode="after")
    def _validate_unit_fields(self) -> "ProviderPricing":
        if self.unit == "money" and not self.currency:
            raise ValueError("money pricing must name its currency")
        if self.unit == "tokens" and self.band is not None:
            raise ValueError("a quota has no time-of-day tier; drop the band")
        periods = [window.period for window in self.windows]
        duplicated = sorted({period for period in periods if periods.count(period) > 1})
        if duplicated:
            raise ValueError(f"one price block declares a window period twice: {', '.join(duplicated)}")
        return self

    @property
    def has_rates(self) -> bool:
        return self.unit == "money" and bool(self.models)

    def pricing_for(self, model: str) -> tuple[str, ModelPricing] | None:
        """The row covering this model: exact id first, then the longest ``prefix*``."""
        row = self.models.get(model)
        if row is not None:
            return model, row
        candidates = [
            (pattern, value)
            for pattern, value in self.models.items()
            if pattern.endswith("*") and model.startswith(pattern[:-1])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: len(item[0]))


_WINDOW = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_day(value: str) -> date:
    stripped = value.strip()
    if _ISO_DAY.match(stripped) is None:
        raise ValueError(f"{value!r} is not a 2026-10-01 date")
    return date.fromisoformat(stripped)


def _holiday_range(entry: str) -> tuple[date, date]:
    """Parse one holiday declaration: a single day, or an inclusive ``start..end``."""
    parts = entry.split("..")
    if len(parts) == 1:
        single = _parse_day(parts[0])
        return single, single
    if len(parts) != 2:
        raise ValueError(f"holiday {entry!r} must be a date or an inclusive 2026-10-01..2026-10-08 range")
    start, end = _parse_day(parts[0]), _parse_day(parts[1])
    if start > end:
        raise ValueError(f"holiday {entry!r} starts after it ends")
    return start, end


def _in_any_range(day: date, entries: list[str]) -> bool:
    return any(start <= day <= end for entry in entries for start, end in [_holiday_range(entry)])


def _in_window(window: str, minutes: int) -> bool:
    match = _WINDOW.match(window)
    if match is None:
        return False
    start = int(match.group(1)) * 60 + int(match.group(2))
    end = int(match.group(3)) * 60 + int(match.group(4))
    if start <= end:
        return start <= minutes < end
    return minutes >= start or minutes < end

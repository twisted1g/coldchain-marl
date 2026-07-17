from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core import config
from core.config import Weather


@dataclass(slots=True)
class DemandSeries:
    day_of_year: np.ndarray
    weekday: np.ndarray
    weather: np.ndarray
    event_multiplier: np.ndarray
    demand_mean: np.ndarray
    demand: np.ndarray


def sample_weather(rng: np.random.Generator) -> Weather:
    weathers = list(Weather)
    probs = np.array([config.WEATHER_PRIORS[w] for w in weathers], dtype=float)
    probs /= probs.sum()
    return weathers[int(rng.choice(len(weathers), p=probs))]


def demand_mean(
    day_of_year: int, weekday: int, weather: Weather, event_mult: float
) -> float:
    season = 1.0 + config.DEMAND_SEASON_AMP * math.sin(
        2.0 * math.pi * day_of_year / config.DAYS_PER_YEAR
    )
    weekday_mult = config.DEMAND_WEEKEND_MULT if weekday >= 5 else 1.0
    weather_mult = config.DEMAND_WEATHER_MULT[weather]
    return (
        config.INVENTORY_DEMAND_MEAN * season * weekday_mult * weather_mult * event_mult
    )


def sample_demand(
    rng: np.random.Generator,
    day_of_year: int,
    weekday: int,
    weather: Weather,
    event_mult: float,
) -> float:
    sigma = config.DEMAND_NOISE_SIGMA
    noise = float(rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma))
    return demand_mean(day_of_year, weekday, weather, event_mult) * noise


def advance_event(
    rng: np.random.Generator, days_left: int, mult: float
) -> tuple[int, float]:
    if days_left > 0:
        days_left -= 1
        return days_left, mult if days_left > 0 else 1.0
    if float(rng.random()) < config.DEMAND_EVENT_PROB:
        lo, hi = config.DEMAND_EVENT_DURATION_RANGE
        duration = int(rng.integers(lo, hi + 1))
        mult_lo, mult_hi = config.DEMAND_EVENT_MULT_RANGE
        return duration, float(rng.uniform(mult_lo, mult_hi))
    return 0, 1.0


def simulate(
    rng: np.random.Generator, day: int, weekday: int, n_days: int
) -> tuple[DemandSeries, int, float]:
    """Roll the demand process forward n_days; returns the series and the
    event state (days_left, multiplier) as of the last simulated day."""
    weathers = list(Weather)

    event_days_left = 0
    event_mult = 1.0

    days = np.empty(n_days, dtype=np.int64)
    weekdays = np.empty(n_days, dtype=np.int64)
    weather_idx = np.empty(n_days, dtype=np.int64)
    event_mults = np.empty(n_days, dtype=np.float64)
    means = np.empty(n_days, dtype=np.float64)
    demands = np.empty(n_days, dtype=np.float64)

    for t in range(n_days):
        weather = sample_weather(rng)
        event_days_left, event_mult = advance_event(rng, event_days_left, event_mult)

        days[t] = day
        weekdays[t] = weekday
        weather_idx[t] = weathers.index(weather)
        event_mults[t] = event_mult
        means[t] = demand_mean(day, weekday, weather, event_mult)
        demands[t] = sample_demand(rng, day, weekday, weather, event_mult)

        day = (day + 1) % config.DAYS_PER_YEAR
        weekday = (weekday + 1) % config.DAYS_PER_WEEK

    series = DemandSeries(
        day_of_year=days,
        weekday=weekdays,
        weather=weather_idx,
        event_multiplier=event_mults,
        demand_mean=means,
        demand=demands,
    )
    return series, event_days_left, event_mult


def generate_series(seed: int, n_days: int) -> DemandSeries:
    rng = np.random.default_rng(seed)
    day = int(rng.integers(0, config.DAYS_PER_YEAR))
    weekday = int(rng.integers(0, config.DAYS_PER_WEEK))
    series, _, _ = simulate(rng, day, weekday, n_days)
    return series


def backfill_history(
    rng: np.random.Generator, end_day_of_year: int, end_weekday: int, n_days: int
) -> tuple[DemandSeries, int, float]:
    """Pre-episode history: n_days ending at (end_day_of_year, end_weekday) inclusive.

    Daily weather is drawn from priors; the returned event state carries over
    into the episode so an active spike survives the reset boundary.
    """
    start_day = (end_day_of_year - n_days + 1) % config.DAYS_PER_YEAR
    start_weekday = (end_weekday - n_days + 1) % config.DAYS_PER_WEEK
    return simulate(rng, start_day, start_weekday, n_days)


def push_history(
    history: DemandSeries,
    day_of_year: int,
    weekday: int,
    weather: Weather,
    event_multiplier: float,
    mean: float,
    demand: float,
) -> None:
    """Shift the rolling window one day left and write the newest day last."""
    values = (
        day_of_year,
        weekday,
        list(Weather).index(weather),
        event_multiplier,
        mean,
        demand,
    )
    for field, value in zip(DemandSeries.__slots__, values, strict=True):
        arr = getattr(history, field)
        arr[:-1] = arr[1:]
        arr[-1] = value

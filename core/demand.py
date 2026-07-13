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


def generate_series(seed: int, n_days: int) -> DemandSeries:
    rng = np.random.default_rng(seed)
    weathers = list(Weather)
    weather_probs = np.array(
        [config.WEATHER_PRIORS[w] for w in weathers], dtype=float
    )
    weather_probs /= weather_probs.sum()

    day = int(rng.integers(0, config.DAYS_PER_YEAR))
    weekday = int(rng.integers(0, config.DAYS_PER_WEEK))
    event_days_left = 0
    event_mult = 1.0

    days = np.empty(n_days, dtype=np.int64)
    weekdays = np.empty(n_days, dtype=np.int64)
    weather_idx = np.empty(n_days, dtype=np.int64)
    event_mults = np.empty(n_days, dtype=np.float64)
    means = np.empty(n_days, dtype=np.float64)
    demands = np.empty(n_days, dtype=np.float64)

    for t in range(n_days):
        weather = weathers[int(rng.choice(len(weathers), p=weather_probs))]
        event_days_left, event_mult = advance_event(rng, event_days_left, event_mult)

        days[t] = day
        weekdays[t] = weekday
        weather_idx[t] = weathers.index(weather)
        event_mults[t] = event_mult
        means[t] = demand_mean(day, weekday, weather, event_mult)
        demands[t] = sample_demand(rng, day, weekday, weather, event_mult)

        day = (day + 1) % config.DAYS_PER_YEAR
        weekday = (weekday + 1) % config.DAYS_PER_WEEK

    return DemandSeries(
        day_of_year=days,
        weekday=weekdays,
        weather=weather_idx,
        event_multiplier=event_mults,
        demand_mean=means,
        demand=demands,
    )

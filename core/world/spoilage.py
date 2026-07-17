from __future__ import annotations

import math
from typing import Protocol

from core.config import (
    DELAY_RISK_FACTOR,
    HUMIDITY_SEVERITY_SCALE,
    KELVIN_OFFSET,
    R_GAS,
    RISK_LABEL_THRESHOLD,
    FruitKey,
)
from core.world.fruits import FruitParams, get_params


class SpoilageModel(Protocol):
    def risk_delta(
        self,
        fruit: FruitKey,
        temperature_c: float,
        humidity: float,
        delay: float = 0.0,
        dt_ticks: float = 1.0,
    ) -> float: ...


class ArrheniusSpoilage:
    def risk_delta(
        self,
        fruit: FruitKey,
        temperature_c: float,
        humidity: float,
        delay: float = 0.0,
        dt_ticks: float = 1.0,
    ) -> float:
        params = get_params(fruit)
        rate_ratio = self._arrhenius_rate(params, temperature_c) / self._arrhenius_rate(
            params, params.optimal_temp_c
        )
        delta = _per_tick_at_optimal(params) * rate_ratio * dt_ticks
        delta += self._chilling_penalty(params, temperature_c) * dt_ticks
        delta += self._humidity_penalty(params, humidity) * dt_ticks
        delta += _per_tick_at_optimal(params) * DELAY_RISK_FACTOR * delay * dt_ticks
        return delta

    @staticmethod
    def _humidity_penalty(params: FruitParams, humidity: float) -> float:
        low, high = params.optimal_humidity_low, params.optimal_humidity_high
        if low <= humidity <= high:
            return 0.0
        distance = (low - humidity) if humidity < low else (humidity - high)
        severity = distance / HUMIDITY_SEVERITY_SCALE
        return _per_tick_at_optimal(params) * severity

    @staticmethod
    def _arrhenius_rate(params: FruitParams, temperature_c: float) -> float:
        t_kelvin = temperature_c + KELVIN_OFFSET
        return params.arrhenius_pre_factor * math.exp(
            -params.arrhenius_activation_energy_j_per_mol / (R_GAS * t_kelvin)
        )

    @staticmethod
    def _chilling_penalty(params: FruitParams, temperature_c: float) -> float:
        if params.chilling_injury_threshold_c is None:
            return 0.0
        if temperature_c >= params.chilling_injury_threshold_c:
            return 0.0
        severity = (params.chilling_injury_threshold_c - temperature_c) / 10.0
        return _per_tick_at_optimal(params) * severity


def _per_tick_at_optimal(params: FruitParams) -> float:
    """Risk accrued per tick under optimal storage: threshold over shelf life."""
    return RISK_LABEL_THRESHOLD / params.base_shelf_life_ticks


def risk_to_label(risk: float, threshold: float = RISK_LABEL_THRESHOLD) -> int:
    return 1 if risk >= threshold else 0

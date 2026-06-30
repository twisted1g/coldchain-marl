from __future__ import annotations

import math
from typing import Protocol

from core.config import KELVIN_OFFSET, R_GAS, RISK_LABEL_THRESHOLD, FruitKey
from core.fruits import FruitParams, get_params


class SpoilageModel(Protocol):
    def risk_delta(
        self,
        fruit: FruitKey,
        temperature_c: float,
        dt_ticks: float = 1.0,
    ) -> float: ...


class ArrheniusSpoilage:
    def risk_delta(
        self,
        fruit: FruitKey,
        temperature_c: float,
        dt_ticks: float = 1.0,
    ) -> float:
        params = get_params(fruit)
        per_tick_at_optimal = RISK_LABEL_THRESHOLD / params.base_shelf_life_ticks
        rate_ratio = self._arrhenius_rate(params, temperature_c) / self._arrhenius_rate(
            params, self._optimal_temp(params)
        )
        delta = per_tick_at_optimal * rate_ratio * dt_ticks
        delta += self._chilling_penalty(params, temperature_c) * dt_ticks
        return delta

    @staticmethod
    def _arrhenius_rate(params: FruitParams, temperature_c: float) -> float:
        t_kelvin = temperature_c + KELVIN_OFFSET
        return params.arrhenius_pre_factor * math.exp(
            -params.arrhenius_activation_energy_j_per_mol / (R_GAS * t_kelvin)
        )

    @staticmethod
    def _optimal_temp(params: FruitParams) -> float:
        return (params.optimal_temp_low_c + params.optimal_temp_high_c) / 2.0

    @staticmethod
    def _chilling_penalty(params: FruitParams, temperature_c: float) -> float:
        if params.chilling_injury_threshold_c is None:
            return 0.0
        if temperature_c >= params.chilling_injury_threshold_c:
            return 0.0
        severity = (params.chilling_injury_threshold_c - temperature_c) / 10.0
        per_tick_at_optimal = RISK_LABEL_THRESHOLD / params.base_shelf_life_ticks
        return per_tick_at_optimal * severity


def risk_to_label(risk: float, threshold: float = RISK_LABEL_THRESHOLD) -> int:
    return 1 if risk >= threshold else 0

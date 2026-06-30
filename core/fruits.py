from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.config import FruitKey


@dataclass(frozen=True, slots=True)
class FruitParams:
    optimal_temp_low_c: float
    optimal_temp_high_c: float
    chilling_injury_threshold_c: float | None
    arrhenius_pre_factor: float
    arrhenius_activation_energy_j_per_mol: float
    base_shelf_life_ticks: int
    ethylene_sensitive: bool


FRUIT_REGISTRY: Final[dict[FruitKey, FruitParams]] = {
    FruitKey.STRAWBERRY: FruitParams(
        optimal_temp_low_c=0.0,
        optimal_temp_high_c=4.0,
        chilling_injury_threshold_c=None,
        arrhenius_pre_factor=1.2e10,
        arrhenius_activation_energy_j_per_mol=70_000.0,
        base_shelf_life_ticks=14,
        ethylene_sensitive=False,
    ),
    FruitKey.BANANA: FruitParams(
        optimal_temp_low_c=13.0,
        optimal_temp_high_c=15.0,
        chilling_injury_threshold_c=13.0,
        arrhenius_pre_factor=5.0e9,
        arrhenius_activation_energy_j_per_mol=65_000.0,
        base_shelf_life_ticks=18,
        ethylene_sensitive=True,
    ),
    FruitKey.ORANGE: FruitParams(
        optimal_temp_low_c=4.0,
        optimal_temp_high_c=8.0,
        chilling_injury_threshold_c=3.0,
        arrhenius_pre_factor=3.0e9,
        arrhenius_activation_energy_j_per_mol=60_000.0,
        base_shelf_life_ticks=28,
        ethylene_sensitive=False,
    ),
}


def get_params(fruit: FruitKey) -> FruitParams:
    return FRUIT_REGISTRY[fruit]

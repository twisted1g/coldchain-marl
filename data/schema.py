from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import FruitKey, Weather
from core.world.noise import Disruption


@dataclass(slots=True)
class StepRecord:
    tick: int
    observations: dict[str, np.ndarray]
    actions: dict[str, Any]
    rewards: dict[str, float]
    spoilage_risk: float
    ground_truth_label: int
    active_disruptions: list[Disruption]
    sensor_temperature_c: float
    energy_usage: float
    current_node: str
    delivered: bool


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: int
    seed: int
    fruit_type: FruitKey
    source_node: str
    target_node: str
    ambient_weather: Weather
    ambient_temp_c: float
    max_steps: int
    steps: list[StepRecord] = field(default_factory=list)
    final_label: int = 0


@dataclass(slots=True)
class DatasetManifest:
    master_seed: int
    n_episodes: int
    library_versions: dict[str, str]
    dataset_sha256: str
    config_snapshot: dict[str, Any]

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from data.schema import EpisodeRecord


class ColdChainDataset:
    def __init__(self, dataset_dir: Path, episodes: list[EpisodeRecord]) -> None:
        self._dir = dataset_dir
        self._episodes = list(episodes)

    @classmethod
    def from_dir(cls, dataset_dir: Path) -> ColdChainDataset:
        dataset_dir = Path(dataset_dir)
        paths = sorted(dataset_dir.glob("episode_*.pkl"))
        if not paths:
            raise FileNotFoundError(f"no episode_*.pkl files found in {dataset_dir}")
        episodes: list[EpisodeRecord] = []
        for path in paths:
            with path.open("rb") as f:
                episodes.append(pickle.load(f))
        return cls(dataset_dir, episodes)

    @property
    def dataset_dir(self) -> Path:
        return self._dir

    @property
    def manifest(self) -> dict[str, Any]:
        path = self._dir / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def episode_ids(self) -> tuple[int, ...]:
        return tuple(ep.episode_id for ep in self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[EpisodeRecord]:
        return iter(self._episodes)

    def __getitem__(self, idx: int) -> EpisodeRecord:
        return self._episodes[idx]

    def subset(self, indices: list[int]) -> ColdChainDataset:
        selected = [self._episodes[i] for i in indices]
        return ColdChainDataset(self._dir, selected)

    def train_val_split(
        self, val_fraction: float = 0.2, seed: int = 0
    ) -> tuple[ColdChainDataset, ColdChainDataset]:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        n = len(self._episodes)
        if n < 2:
            raise ValueError(f"need at least 2 episodes to split, got {n}")
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_val = max(1, min(n - 1, int(round(n * val_fraction))))
        val_idx = sorted(indices[:n_val].tolist())
        train_idx = sorted(indices[n_val:].tolist())
        return self.subset(train_idx), self.subset(val_idx)

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from core import config
from core.demand import DemandSeries, generate_series
from data.generate import _config_snapshot, _resolved_versions, derive_seed

N_SERIES_DEFAULT: Final[int] = 50
N_DAYS_DEFAULT: Final[int] = 2000
TRAIN_FRACTION: Final[float] = 0.7
VAL_FRACTION: Final[float] = 0.15

_FIELDS: Final[tuple[str, ...]] = (
    "day_of_year",
    "weekday",
    "weather",
    "event_multiplier",
    "demand_mean",
    "demand",
)


def _stack(series: list[DemandSeries]) -> dict[str, np.ndarray]:
    return {field: np.stack([getattr(s, field) for s in series]) for field in _FIELDS}


def _split_bounds(n_days: int) -> dict[str, list[int]]:
    train_end = int(n_days * TRAIN_FRACTION)
    val_end = int(n_days * (TRAIN_FRACTION + VAL_FRACTION))
    return {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, n_days],
    }


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for field in _FIELDS:
        h.update(arrays[field].tobytes())
    return h.hexdigest()


def generate_demand(
    master_seed: int, n_series: int, n_days: int, out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [derive_seed(master_seed, i) for i in range(n_series)]
    arrays = _stack([generate_series(seed, n_days) for seed in seeds])
    np.savez_compressed(out_dir / "demand.npz", **arrays)

    manifest = {
        "master_seed": master_seed,
        "n_series": n_series,
        "n_days": n_days,
        "series_seeds": seeds,
        "fields": list(_FIELDS),
        "split": _split_bounds(n_days),
        "dataset_sha256": _hash_arrays(arrays),
        "library_versions": _resolved_versions(),
        "config_snapshot": _config_snapshot(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def load_demand(dataset_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    with np.load(dataset_dir / "demand.npz") as data:
        arrays = {field: data[field] for field in manifest["fields"]}
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exogenous demand series for the forecaster."
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--series", type=int, default=N_SERIES_DEFAULT)
    parser.add_argument("--days", type=int, default=N_DAYS_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path("data/demand"))
    args = parser.parse_args()
    manifest = generate_demand(args.seed, args.series, args.days, args.out)
    print(
        f"Generated {manifest['n_series']} series × {manifest['n_days']} days → {args.out}"
    )
    print(f"split: {manifest['split']}")
    print(f"dataset_sha256={manifest['dataset_sha256'][:16]}...")


if __name__ == "__main__":
    main()

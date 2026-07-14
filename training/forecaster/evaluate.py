from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core import config
from data.generate_demand import DEMAND_DIR, load_demand
from training.config import FORECASTER_PATH
from training.forecaster.model import WINDOW, build_features, load_forecaster, make_windows

BATCH_SIZE = 1024


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(pred - y).mean()),
        "mape": float((np.abs(pred - y) / y).mean() * 100.0),
    }


@torch.no_grad()
def _predict(model, x: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, len(x), BATCH_SIZE):
        preds.append(model(torch.as_tensor(x[i : i + BATCH_SIZE])).numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the demand forecaster against naive baselines "
        "and the generator noise floor on the test split."
    )
    parser.add_argument("--data", type=Path, default=DEMAND_DIR)
    parser.add_argument("--checkpoint", type=Path, default=FORECASTER_PATH)
    args = parser.parse_args()

    arrays, manifest = load_demand(args.data)
    t_start, t_end = manifest["split"]["test"]
    lo = max(t_start, WINDOW)

    features = build_features(arrays)
    x, y = make_windows(features, t_start, t_end)
    model = load_forecaster(args.checkpoint)
    pred = _predict(model, x)

    scale = config.INVENTORY_DEMAND_MEAN
    demand = arrays["demand"] / scale
    oracle = (arrays["demand_mean"] / scale)[:, lo:t_end].reshape(-1)
    last_value = demand[:, lo - 1 : t_end - 1].reshape(-1)
    seasonal = demand[:, lo - config.DAYS_PER_WEEK : t_end - config.DAYS_PER_WEEK]
    event_day = (arrays["event_multiplier"][:, lo:t_end] > 1.0).reshape(-1)

    rows = {
        "transformer": pred,
        "last_value": last_value,
        "seasonal_naive_t-7": seasonal.reshape(-1),
        "oracle_floor": oracle,
    }
    print(f"test targets: {len(y)}  event days: {int(event_day.sum())}")
    print(
        f"{'predictor':<20}{'mae':>8}{'mape%':>8}{'mae@event':>11}{'mape%@event':>13}"
    )
    for name, p in rows.items():
        m = _metrics(p, y)
        e = _metrics(p[event_day], y[event_day])
        print(
            f"{name:<20}{m['mae']:>8.4f}{m['mape']:>8.2f}{e['mae']:>11.4f}{e['mape']:>13.2f}"
        )

    lv_mae = _metrics(last_value, y)["mae"]
    model_mae = _metrics(pred, y)["mae"]
    floor_mae = _metrics(oracle, y)["mae"]
    extracted = (lv_mae - model_mae) / (lv_mae - floor_mae) * 100.0
    print(
        f"\npredictable signal extracted: {extracted:.1f}% "
        f"(last_value {lv_mae:.4f} -> transformer {model_mae:.4f}, "
        f"oracle floor {floor_mae:.4f})"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core import config
from core.config import Weather
from core.demand import DemandSeries

WINDOW = config.DEMAND_HISTORY_DAYS
SEQ_LEN = WINDOW + 1
N_WEATHER = len(Weather)
N_FEATURES = 6 + N_WEATHER

D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


def build_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Per-day feature channels:
    [demand_norm, event_mult-1, sin/cos year, sin/cos week, weather one-hot].

    Demand is normalized by the base mean; channel 0 doubles as the target series.
    Event channel is ex-post history — the target-day token zeroes it out.
    """
    angle_year = 2.0 * np.pi * arrays["day_of_year"] / config.DAYS_PER_YEAR
    angle_week = 2.0 * np.pi * arrays["weekday"] / config.DAYS_PER_WEEK
    weather = np.eye(N_WEATHER, dtype=np.float64)[arrays["weather"]]
    return np.concatenate(
        [
            (arrays["demand"] / config.INVENTORY_DEMAND_MEAN)[..., None],
            (arrays["event_multiplier"] - 1.0)[..., None],
            np.sin(angle_year)[..., None],
            np.cos(angle_year)[..., None],
            np.sin(angle_week)[..., None],
            np.cos(angle_week)[..., None],
            weather,
        ],
        axis=-1,
    ).astype(np.float32)


def make_windows(
    features: np.ndarray, t_start: int, t_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows over [series, days, features] -> (X [n, SEQ_LEN, F], y [n]).

    Sequence = 28 history days + a target-day token carrying the known covariates
    of day t (calendar deterministic, weather as forecast) with demand and event
    channels zeroed. Targets t range over [max(t_start, WINDOW), t_end); inputs
    may reach back across the split boundary — history-only, no target leakage.
    """
    lo = max(t_start, WINDOW)
    x = np.stack([features[:, t - WINDOW : t] for t in range(lo, t_end)], axis=1)
    target_token = features[:, lo:t_end].copy()
    target_token[..., :2] = 0.0
    x = np.concatenate([x, target_token[:, :, None, :]], axis=2)
    y = features[:, lo:t_end, 0]
    return x.reshape(-1, SEQ_LEN, features.shape[-1]), y.reshape(-1).copy()


class DemandForecaster(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = N_LAYERS,
        window: int = WINDOW,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(window + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_embedding
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


@torch.no_grad()
def predict_next(
    model: DemandForecaster,
    history: DemandSeries,
    day_of_year: int,
    weekday: int,
    weather: Weather,
) -> float:
    """One-step forecast from a rolling history window + target-day covariates.

    Returns demand in raw units (denormalized)."""
    arrays = {field: getattr(history, field)[None] for field in DemandSeries.__slots__}
    features = build_features(arrays)

    angle_year = 2.0 * np.pi * day_of_year / config.DAYS_PER_YEAR
    angle_week = 2.0 * np.pi * weekday / config.DAYS_PER_WEEK
    target_token = np.zeros((1, 1, N_FEATURES), dtype=np.float32)
    target_token[0, 0, 2:6] = (
        np.sin(angle_year),
        np.cos(angle_year),
        np.sin(angle_week),
        np.cos(angle_week),
    )
    target_token[0, 0, 6 + list(Weather).index(weather)] = 1.0

    x = np.concatenate([features, target_token], axis=1)
    pred = float(model(torch.as_tensor(x)))
    return max(0.0, pred * config.INVENTORY_DEMAND_MEAN)


def load_forecaster(path) -> DemandForecaster:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DemandForecaster(
        n_features=checkpoint["n_features"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        window=checkpoint["window"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

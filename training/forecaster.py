from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core import config
from core.config import Weather

WINDOW = 28
N_WEATHER = len(Weather)
N_FEATURES = 5 + N_WEATHER

D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


def build_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Per-day feature channels: [demand_norm, sin/cos year, sin/cos week, weather one-hot].

    Demand is normalized by the base mean; channel 0 doubles as the target series.
    """
    angle_year = 2.0 * np.pi * arrays["day_of_year"] / config.DAYS_PER_YEAR
    angle_week = 2.0 * np.pi * arrays["weekday"] / config.DAYS_PER_WEEK
    weather = np.eye(N_WEATHER, dtype=np.float64)[arrays["weather"]]
    return np.concatenate(
        [
            (arrays["demand"] / config.INVENTORY_DEMAND_MEAN)[..., None],
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
    """Sliding windows over [series, days, features] -> (X [n, WINDOW, F], y [n]).

    Targets t range over [max(t_start, WINDOW), t_end); inputs may reach back
    across the split boundary — history-only, no target leakage.
    """
    lo = max(t_start, WINDOW)
    x = np.stack([features[:, t - WINDOW : t] for t in range(lo, t_end)], axis=1)
    y = features[:, lo:t_end, 0]
    return x.reshape(-1, WINDOW, features.shape[-1]), y.reshape(-1).copy()


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
        self.pos_embedding = nn.Parameter(torch.zeros(window, d_model))
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

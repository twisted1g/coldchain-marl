from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core import config
from data.generate_demand import load_demand
from training.config import FORECASTER_PATH
from training.forecaster import (
    D_MODEL,
    N_FEATURES,
    N_HEAD,
    N_LAYERS,
    WINDOW,
    DemandForecaster,
    build_features,
    make_windows,
)

SEED = 0
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


def _loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def _evaluate(model: DemandForecaster, loader: DataLoader) -> dict[str, float]:
    model.eval()
    abs_err = sq_err = last_abs_err = n = 0.0
    for x, y in loader:
        pred = model(x)
        abs_err += float((pred - y).abs().sum())
        sq_err += float(((pred - y) ** 2).sum())
        last_abs_err += float((x[:, -1, 0] - y).abs().sum())
        n += len(y)
    return {"mae": abs_err / n, "mse": sq_err / n, "last_value_mae": last_abs_err / n}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the demand transformer on the exogenous dataset."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data", type=Path, default=Path("data/demand"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays, manifest = load_demand(args.data)
    features = build_features(arrays)
    split = manifest["split"]
    x_train, y_train = make_windows(features, *split["train"])
    x_val, y_val = make_windows(features, *split["val"])
    print(f"windows: train {len(y_train)}, val {len(y_val)}")

    train_loader = _loader(x_train, y_train, shuffle=True)
    val_loader = _loader(x_val, y_val, shuffle=False)

    model = DemandForecaster()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y)
        metrics = _evaluate(model, val_loader)
        if metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch:3d}  loss={epoch_loss / len(y_train):.5f}  "
            f"val_mae={metrics['mae']:.4f}  val_last_value_mae="
            f"{metrics['last_value_mae']:.4f}"
        )

    FORECASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "n_features": N_FEATURES,
            "d_model": D_MODEL,
            "nhead": N_HEAD,
            "num_layers": N_LAYERS,
            "window": WINDOW,
            "dataset_sha256": manifest["dataset_sha256"],
            "demand_scale": config.INVENTORY_DEMAND_MEAN,
        },
        FORECASTER_PATH,
    )
    print(f"best val_mae={best_val_mae:.4f} (normalized units)")
    print(f"saved forecaster -> {FORECASTER_PATH}")


if __name__ == "__main__":
    main()

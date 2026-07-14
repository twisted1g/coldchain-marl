from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from core.dynamics import step
from core.world.graph_features import spoilage_node_features, static_edge_index
from core.interfaces.spaces import ACTION_SPACES
from core.world.spoilage import risk_to_label
from core.state import init_state
from training.config import SPOILAGE_ENCODER_PATH
from training.marl.gnn import SpoilagePretrainModel

SEED = 0
N_EPISODES = 1200
EPISODE_MAX_STEPS = 20
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-3
VAL_FRACTION = 0.2


def _build_samples(seed: int, n_episodes: int) -> list[Data]:
    rng = np.random.default_rng(seed)
    spaces = dict(ACTION_SPACES)
    for space in spaces.values():
        space.seed(int(rng.integers(0, 2**31 - 1)))

    samples: list[Data] = []
    for ep in range(n_episodes):
        state = init_state(seed=seed + ep, max_steps=EPISODE_MAX_STEPS)
        edge_index = torch.as_tensor(static_edge_index(state.graph), dtype=torch.long)
        for _ in range(state.max_steps):
            actions = {name: space.sample() for name, space in spaces.items()}
            result = step(state, actions)
            x = torch.as_tensor(spoilage_node_features(state), dtype=torch.float32)
            y = float(risk_to_label(state.shipment.spoilage_risk))
            samples.append(Data(x=x, edge_index=edge_index, y=torch.tensor([y])))
            if result.terminated["__all__"]:
                break
    return samples


@torch.no_grad()
def _evaluate(model: SpoilagePretrainModel, loader: DataLoader) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.batch)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        y = batch.y.view(-1)
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    fn_rate = fn / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": acc,
        "fn_rate": fn_rate,
        "pos_frac": (tp + fn) / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline pretrain of the spoilage GNN encoder."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = _build_samples(args.seed, args.episodes)
    n_val = max(1, int(len(samples) * VAL_FRACTION))
    perm = np.random.default_rng(args.seed).permutation(len(samples))
    val_idx, train_idx = set(perm[:n_val].tolist()), perm[n_val:].tolist()
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in sorted(val_idx)]

    pos = sum(float(s.y.item()) for s in train)
    pos_weight = (
        torch.tensor([(len(train) - pos) / pos]) if pos else torch.tensor([1.0])
    )
    print(
        f"samples: {len(samples)} (train {len(train)}, val {len(val)})  "
        f"pos_frac={pos / len(train):.3f}"
    )

    model = SpoilagePretrainModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.num_graphs
        if epoch % 5 == 0 or epoch == args.epochs:
            m = _evaluate(model, val_loader)
            print(
                f"epoch {epoch:3d}  loss={epoch_loss / len(train):.4f}  "
                f"val_acc={m['accuracy']:.3f}  val_fn_rate={m['fn_rate']:.3f}"
            )

    SPOILAGE_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), SPOILAGE_ENCODER_PATH)
    print(f"saved frozen encoder -> {SPOILAGE_ENCODER_PATH}")


if __name__ == "__main__":
    main()

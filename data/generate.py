from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import pickle
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from core import config
from core.dynamics import step
from core.spaces import ACTION_SPACES
from core.state import init_state
from data.schema import DatasetManifest, EpisodeRecord, StepRecord


def derive_seed(master_seed: int, episode_id: int) -> int:
    h = hashlib.sha256(f"{master_seed}:{episode_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def generate_episode(master_seed: int, episode_id: int) -> EpisodeRecord:
    seed = derive_seed(master_seed, episode_id)
    state = init_state(seed=seed)

    action_rng = np.random.default_rng(seed ^ 0xA17104)
    sampled_action_spaces = {name: space for name, space in ACTION_SPACES.items()}
    for space in sampled_action_spaces.values():
        space.seed(int(action_rng.integers(0, 2**31 - 1)))

    episode = EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        fruit_type=state.shipment.fruit_type,
        source_node=state.shipment.current_node,
        target_node=state.shipment.target_node,
        ambient_weather=state.ambient_weather,
        ambient_temp_c=state.ambient_temp_c,
        max_steps=state.max_steps,
    )

    for _ in range(state.max_steps):
        actions = {name: space.sample() for name, space in sampled_action_spaces.items()}
        result = step(state, actions)
        episode.steps.append(
            StepRecord(
                tick=state.tick,
                observations={k: v.copy() for k, v in result.observations.items()},
                actions=actions,
                rewards=dict(result.rewards),
                spoilage_risk=state.shipment.spoilage_risk,
                ground_truth_label=state.shipment.ground_truth_label,
                active_disruptions=list(state.active_disruptions),
                sensor_temperature_c=state.shipment.sensor_temperature_c,
                energy_usage=state.energy_usage,
                current_node=state.shipment.current_node,
                delivered=state.shipment.current_node == state.shipment.target_node,
            )
        )
        if result.terminated["__all__"]:
            break

    episode.final_label = state.shipment.ground_truth_label
    return episode


def _config_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if isinstance(value, (int, float, str, bool)):
            snapshot[name] = value
        elif isinstance(value, tuple) and all(isinstance(x, (int, float, str)) for x in value):
            snapshot[name] = list(value)
    return snapshot


_TRACKED_PACKAGES: tuple[str, ...] = (
    "gymnasium",
    "networkx",
    "numpy",
    "pettingzoo",
    "tensordict",
    "torch",
    "torch-geometric",
    "torchrl",
)


def _resolved_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _hash_episodes(episodes: list[EpisodeRecord]) -> str:
    h = hashlib.sha256()
    for ep in episodes:
        h.update(pickle.dumps(ep))
    return h.hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def generate(master_seed: int, n_episodes: int, out_dir: Path) -> DatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[EpisodeRecord] = []
    for episode_id in range(n_episodes):
        ep = generate_episode(master_seed, episode_id)
        with (out_dir / f"episode_{episode_id:06d}.pkl").open("wb") as f:
            pickle.dump(ep, f)
        episodes.append(ep)

    manifest = DatasetManifest(
        master_seed=master_seed,
        n_episodes=n_episodes,
        library_versions=_resolved_versions(),
        dataset_sha256=_hash_episodes(episodes),
        config_snapshot=_config_snapshot(),
    )
    manifest_dict = _to_jsonable(asdict(manifest)) if is_dataclass(manifest) else {}
    (out_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cold-chain episodes.")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=config.N_EPISODES_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path("./dataset"))
    args = parser.parse_args()
    manifest = generate(args.seed, args.episodes, args.out)
    print(f"Generated {manifest.n_episodes} episodes → {args.out}")
    print(f"dataset_sha256={manifest.dataset_sha256[:16]}...")


if __name__ == "__main__":
    main()
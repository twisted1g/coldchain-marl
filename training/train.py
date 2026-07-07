from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from core.config import DELIVERY_AGENTS
from env.training_env import ColdChainTrainingEnv
from training.agents import RandomAgent
from training.config import (
    ARTIFACTS,
    COMPARE_SEED,
    CURVE_CSV,
    EPISODES_PER_ITERATION,
    EVAL_EPISODES,
    EVAL_SEED,
    LEARNERS,
    METRIC,
    MODULES_DIR,
    NUM_ITERATIONS,
    SEED,
    TRAIN_SEED,
    build_agents,
    env_config,
    module_dir,
)
from training.evaluate import rollout
from training.loop import collect_and_learn


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cold-chain agents (CTDE loop).")
    p.add_argument(
        "--agents",
        nargs="+",
        default=LEARNERS,
        help="learners to train; rest stay frozen",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    learners = args.agents
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ARTIFACTS.mkdir(exist_ok=True)
    MODULES_DIR.mkdir(parents=True, exist_ok=True)

    train_env = ColdChainTrainingEnv(env_config(TRAIN_SEED, learners))
    eval_env = ColdChainTrainingEnv(env_config(EVAL_SEED, learners))
    agents = build_agents(train_env, learners)

    fieldnames = ["iteration"]
    for a in learners:
        fieldnames += [f"return_{a}", f"{METRIC[a][0]}_{a}"]

    rows: list[dict[str, float]] = []
    for it in range(1, NUM_ITERATIONS + 1):
        collect_and_learn(train_env, agents, EPISODES_PER_ITERATION)
        row: dict[str, float] = {"iteration": it}
        parts = []
        for a in learners:
            metric_key = METRIC[a][0]
            ret, metric = rollout(eval_env, agents, a, EVAL_EPISODES, metric_key)
            row[f"return_{a}"] = ret
            row[f"{metric_key}_{a}"] = metric
            parts.append(f"{a}: return={ret:8.3f} {metric_key}={metric:7.3f}")
        rows.append(row)
        print(f"iter {it:3d}  " + "  |  ".join(parts))

    with CURVE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for a in learners:
        agents[a].save(module_dir(a))

    _compare(learners)
    print(f"\nsaved curve -> {CURVE_CSV}\nsaved modules -> {MODULES_DIR}")


def _print_compare(
    name: str, metric_key: str, direction: str, trained_m: float, random_m: float
) -> None:
    better = (random_m - trained_m) if direction == "min" else (trained_m - random_m)
    margin = better / abs(random_m) if random_m else float("nan")
    print(
        f"  {name}: trained {metric_key}={trained_m:.3f}  "
        f"random={random_m:.3f}  ({margin:+.0%})"
    )


def _compare(learners: list[str]) -> None:
    """Trained-vs-random sanity check per learner block on a held-out seed set."""
    print("\ntrained vs random:")
    for a in learners:
        if a not in DELIVERY_AGENTS:
            _compare_block(a, [a])
    delivery = [a for a in learners if a in DELIVERY_AGENTS]
    if delivery:
        _compare_block("delivery", delivery)


def _compare_block(name: str, block: list[str]) -> None:
    """Compare one learner block (single agent, or delivery MADDPG vehicle group)."""
    metric_key, direction = METRIC[block[0]]
    env = ColdChainTrainingEnv(env_config(COMPARE_SEED, block))

    trained = build_agents(env, block)
    trained[block[0]].load(module_dir(block[0]))
    trained_m = float(
        np.mean([rollout(env, trained, a, EVAL_EPISODES, metric_key)[1] for a in block])
    )

    rand = build_agents(env, block)
    for a in block:
        rand[a] = RandomAgent(env.action_space(a))
    random_m = float(
        np.mean([rollout(env, rand, a, EVAL_EPISODES, metric_key)[1] for a in block])
    )

    _print_compare(name, metric_key, direction, trained_m, random_m)


if __name__ == "__main__":
    main()

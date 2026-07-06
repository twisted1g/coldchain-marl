from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from env.training_env import ColdChainTrainingEnv
from training.agents import RandomAgent
from training.config import (
    ARTIFACTS,
    COMPARE_SEED,
    CURVE_CSV,
    EVAL_EPISODES,
    EVAL_SEED,
    EPISODES_PER_ITERATION,
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
    p.add_argument("--agents", nargs="+", default=LEARNERS, help="learners to train; rest stay frozen")
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


def _compare(learners: list[str]) -> None:
    """Trained-vs-random sanity check per learner on a held-out seed set."""
    print("\ntrained vs random:")
    for a in learners:
        metric_key, direction = METRIC[a]
        # Ceteris-paribus: measure the primary agent against a FIXED frozen (no-op) backdrop,
        # swapping only the primary trained-vs-random. Co-learners must be identical in both
        # arms; a frozen no-op backdrop is deterministic AND keeps the world non-trivial
        # (e.g. leaves temperature uncontrolled so spoilage risk exists to predict). Loading
        # co-learners as trained suppresses that risk; loading them untrained is nondeterministic.
        solo = [a]
        env = ColdChainTrainingEnv(env_config(COMPARE_SEED, solo))
        trained = build_agents(env, solo)
        trained[a].load(module_dir(a))
        _, trained_m = rollout(env, trained, a, EVAL_EPISODES, metric_key)

        rand = build_agents(env, solo)
        rand[a] = RandomAgent(env.action_space(a))
        _, random_m = rollout(env, rand, a, EVAL_EPISODES, metric_key)

        better = (random_m - trained_m) if direction == "min" else (trained_m - random_m)
        margin = better / abs(random_m) if random_m else float("nan")
        print(f"  {a}: trained {metric_key}={trained_m:.3f}  random={random_m:.3f}  ({margin:+.0%})")


if __name__ == "__main__":
    main()

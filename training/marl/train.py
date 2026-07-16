from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from core.config import DELIVERY_AGENTS
from env.training_env import ColdChainTrainingEnv
from training.config import (
    ARTIFACTS,
    COMPARE_METRIC,
    COMPARE_SEED,
    CURVE_CSV,
    EPISODES_PER_ITERATION,
    EVAL_EPISODES,
    EVAL_SEED,
    FORECASTER_PATH,
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
from training.marl.agents import RandomAgent
from training.marl.evaluate import rollout
from training.marl.loop import collect_and_learn


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cold-chain agents (CTDE loop).")
    p.add_argument(
        "--agents",
        nargs="+",
        default=LEARNERS,
        help="learners to train; rest stay frozen",
    )
    p.add_argument(
        "--forecaster",
        action="store_true",
        help="fill demand_forecast with the frozen transformer (default: stub)",
    )
    p.add_argument(
        "--load",
        nargs="+",
        default=[],
        help="learners to warm-start from saved modules (fine-tune)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="suffix for the reward-curve csv (ablation runs don't clobber)",
    )
    p.add_argument(
        "--scenario-bank",
        nargs="?",
        const="data/scenarios/bank.json",
        default=None,
        help="train (and eval) with LLM disruption scenarios replayed per episode",
    )
    p.add_argument(
        "--scenario-prob",
        type=float,
        default=1.0,
        help="probability an episode replays a scenario (rest stay clean)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    learners = args.agents
    forecaster = FORECASTER_PATH if args.forecaster else None
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ARTIFACTS.mkdir(exist_ok=True)
    MODULES_DIR.mkdir(parents=True, exist_ok=True)

    train_env = ColdChainTrainingEnv(
        env_config(TRAIN_SEED, learners, forecaster, args.scenario_bank, args.scenario_prob)
    )
    eval_env = ColdChainTrainingEnv(
        env_config(EVAL_SEED, learners, forecaster, args.scenario_bank, args.scenario_prob)
    )
    agents = build_agents(train_env, learners)
    for a in args.load:
        agents[a].load(module_dir(a))

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

    curve_csv = (
        CURVE_CSV.with_stem(f"{CURVE_CSV.stem}_{args.tag}") if args.tag else CURVE_CSV
    )
    with curve_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for a in learners:
        save_dir = MODULES_DIR / f"{a}_{args.tag}" if args.tag else module_dir(a)
        agents[a].save(save_dir)

    _compare(learners, forecaster, args.tag)
    print(f"\nsaved curve -> {curve_csv}\nsaved modules -> {MODULES_DIR}")


def _print_compare(
    name: str, metric_key: str, direction: str, trained_m: float, random_m: float
) -> None:
    better = (random_m - trained_m) if direction == "min" else (trained_m - random_m)
    # Percent margin is noise when the random baseline is near zero.
    if abs(random_m) >= 0.05:
        margin = f"{better / abs(random_m):+.0%}"
    else:
        margin = f"{better:+.3f} abs"
    print(
        f"  {name}: trained {metric_key}={trained_m:.3f}  "
        f"random={random_m:.3f}  ({margin})"
    )


def _compare(learners: list[str], forecaster=None, tag: str | None = None) -> None:
    """Trained-vs-random sanity check per learner block on a held-out seed set."""
    print("\ntrained vs random:")
    for a in learners:
        if a not in DELIVERY_AGENTS:
            _compare_block(a, [a], forecaster, tag)
    delivery = [a for a in learners if a in DELIVERY_AGENTS]
    if delivery:
        _compare_block("delivery", delivery, forecaster, tag)


def _compare_block(
    name: str, block: list[str], forecaster=None, tag: str | None = None
) -> None:
    """Compare one learner block (single agent, or delivery MADDPG vehicle group)."""
    metric_key, direction = COMPARE_METRIC[block[0]]
    env = ColdChainTrainingEnv(env_config(COMPARE_SEED, block, forecaster))

    trained = build_agents(env, block)
    load_dir = (
        MODULES_DIR / f"{block[0]}_{tag}" if tag else module_dir(block[0])
    )
    trained[block[0]].load(load_dir)
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

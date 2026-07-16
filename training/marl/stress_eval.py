"""Stress-test trained agents against the LLM disruption-scenario bank.

For every learner block, compares the COMPARE_METRIC on clean episodes
against episodes replaying each scenario category from the bank
(paper Section 5: quantitative resilience evaluation under prescribed
disruption conditions).

Usage:
    uv run python -m training.marl.stress_eval [--forecaster] [--episodes 3]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from core.config import DELIVERY_AGENTS
from env.training_env import ColdChainTrainingEnv
from llm.scenarios import load_bank
from training.config import (
    COMPARE_METRIC,
    FORECASTER_PATH,
    LEARNERS,
    SEED,
    env_config,
    load_agents,
)
from training.marl.evaluate import rollout

STRESS_SEED = 700_000
DEFAULT_BANK = "data/scenarios/bank.json"
DEFAULT_EPISODES_PER_SCENARIO = 3


def _blocks(learners: list[str]) -> dict[str, list[str]]:
    blocks = {a: [a] for a in learners if a not in DELIVERY_AGENTS}
    delivery = [a for a in learners if a in DELIVERY_AGENTS]
    if delivery:
        blocks["delivery"] = delivery
    return blocks


def _block_metric(
    env: ColdChainTrainingEnv,
    agents: dict,
    block: list[str],
    metric_key: str,
    n_episodes: int,
    reset_options: dict | None = None,
) -> float:
    return float(
        np.mean(
            [
                rollout(env, agents, a, n_episodes, metric_key, reset_options)[1]
                for a in block
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--forecaster", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES_PER_SCENARIO,
        help="episodes per scenario (and per-category count for the clean run)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="load suffixed module variants (e.g. scenario-fine-tuned)",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    forecaster = FORECASTER_PATH if args.forecaster else None

    by_category: dict[str, list[str]] = defaultdict(list)
    for scenario in load_bank(args.bank):
        by_category[scenario.category].append(scenario.id)
    categories = sorted(by_category)

    results: dict[str, dict[str, float]] = {}
    for name, block in _blocks(LEARNERS).items():
        metric_key, direction = COMPARE_METRIC[block[0]]
        clean_env = ColdChainTrainingEnv(env_config(STRESS_SEED, block, forecaster))
        agents = load_agents(clean_env, block, args.tag)

        clean_episodes = args.episodes * max(len(v) for v in by_category.values())
        row = {"clean": _block_metric(clean_env, agents, block, metric_key, clean_episodes)}

        stress_cfg = env_config(STRESS_SEED, block, forecaster)
        stress_cfg["scenario_bank"] = args.bank
        stress_env = ColdChainTrainingEnv(stress_cfg)
        for category in categories:
            values = [
                _block_metric(
                    stress_env, agents, block, metric_key, args.episodes,
                    reset_options={"scenario_id": sid},
                )
                for sid in by_category[category]
            ]
            row[category] = float(np.mean(values))
        results[f"{name} ({metric_key},{direction})"] = row

    _print_table(results, categories)


def _print_table(results: dict[str, dict[str, float]], categories: list[str]) -> None:
    col_names = ["clean", *categories]
    width = max(len(c) for c in col_names) + 2
    label_width = max(len(name) for name in results) + 2
    print("\nmetric per block: clean vs scenario category")
    print(" " * label_width + "".join(f"{c:>{width}}" for c in col_names))
    for name, row in results.items():
        cells = "".join(f"{row[c]:>{width}.3f}" for c in col_names)
        print(f"{name:<{label_width}}" + cells)

    print("\ndegradation vs clean (positive = worse):")
    print(" " * label_width + "".join(f"{c:>{width}}" for c in categories))
    for name, row in results.items():
        direction = name.rsplit(",", 1)[-1].rstrip(")")
        sign = 1.0 if direction == "min" else -1.0
        cells = ""
        for c in categories:
            delta = sign * (row[c] - row["clean"])
            base = abs(row["clean"])
            cells += (
                f"{delta / base:>{width - 1}.0%} " if base >= 0.05
                else f"{delta:>{width - 4}.3f}abs "
            )
        print(f"{name:<{label_width}}" + cells)


if __name__ == "__main__":
    main()

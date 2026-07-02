from __future__ import annotations

import csv

import ray

from training.config import (
    ARTIFACTS,
    CURVE_CSV,
    EVAL_ENV_CONFIG,
    EVAL_EPISODES,
    LEARNERS,
    MODULES_DIR,
    NUM_ITERATIONS,
    build_config,
    module_dir,
)
from training.env import ColdChainTrainingEnv
from training.evaluate import greedy_action_fn, rollout


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        algo = build_config().build_algo()
        eval_env = ColdChainTrainingEnv(EVAL_ENV_CONFIG)
        primary = LEARNERS[0]
        action_space = eval_env.action_space(primary)

        rows: list[dict[str, float]] = []
        for it in range(1, NUM_ITERATIONS + 1):
            algo.train()
            mean_return, mean_dev = rollout(
                eval_env,
                greedy_action_fn(algo.get_module(primary), action_space, primary),
                EVAL_EPISODES,
                primary,
            )
            rows.append({"iteration": it, "mean_return": mean_return, "mean_temp_deviation": mean_dev})
            print(f"iter {it:3d}  return={mean_return:8.3f}  temp_dev={mean_dev:7.3f}")

        with CURVE_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "mean_return", "mean_temp_deviation"])
            writer.writeheader()
            writer.writerows(rows)
        for agent in LEARNERS:
            algo.get_module(agent).save_to_path(module_dir(agent))
        print(f"\nsaved curve -> {CURVE_CSV}\nsaved modules -> {MODULES_DIR}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

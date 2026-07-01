from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ray

from training.config import (
    ARTIFACTS,
    CURVE_CSV,
    CURVE_PNG,
    EVAL_ENV_CONFIG,
    EVAL_EPISODES,
    LEARNER,
    MODULE_DIR,
    NUM_ITERATIONS,
    build_config,
)
from training.env import TemperatureTrainingEnv
from training.evaluate import greedy_action_fn, rollout


def save_plot(rows: list[dict[str, float]]) -> None:
    iters = [r["iteration"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(iters, [r["mean_return"] for r in rows], marker="o", ms=3, color="tab:blue")
    ax1.set(title="Mean episode return", xlabel="iteration", ylabel="greedy return")
    ax1.grid(True, alpha=0.3)
    ax2.plot(iters, [r["mean_temp_deviation"] for r in rows], marker="o", ms=3, color="tab:red")
    ax2.set(title="Mean |temp - ideal| (°C)", xlabel="iteration", ylabel="deviation")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=120)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        algo = build_config().build_algo()
        eval_env = TemperatureTrainingEnv(EVAL_ENV_CONFIG)
        action_space = eval_env.action_space(LEARNER)

        rows: list[dict[str, float]] = []
        for it in range(1, NUM_ITERATIONS + 1):
            algo.train()
            mean_return, mean_dev = rollout(
                eval_env, greedy_action_fn(algo.get_module(LEARNER), action_space), EVAL_EPISODES
            )
            rows.append({"iteration": it, "mean_return": mean_return, "mean_temp_deviation": mean_dev})
            print(f"iter {it:3d}  return={mean_return:8.3f}  temp_dev={mean_dev:7.3f}")

        with CURVE_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "mean_return", "mean_temp_deviation"])
            writer.writeheader()
            writer.writerows(rows)
        save_plot(rows)
        algo.get_module(LEARNER).save_to_path(MODULE_DIR)
        print(f"\nsaved curve -> {CURVE_CSV}\nsaved plot -> {CURVE_PNG}\nsaved module -> {MODULE_DIR}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

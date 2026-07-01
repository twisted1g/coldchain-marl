from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from data.schema import EpisodeRecord


def load_episodes(dataset_dir: Path) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    for path in sorted(dataset_dir.glob("episode_*.pkl")):
        with path.open("rb") as f:
            episodes.append(pickle.load(f))
    return episodes


def _fmt_pct(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else f"{count} (—)"


def summarize_manifest(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print("[manifest] not found")
        return
    manifest = json.loads(manifest_path.read_text())
    print("[manifest]")
    print(f"  master_seed={manifest.get('master_seed')}")
    print(f"  n_episodes={manifest.get('n_episodes')}")
    print(f"  dataset_sha256={manifest.get('dataset_sha256', '')[:16]}...")
    versions = manifest.get("library_versions", {})
    if versions:
        joined = ", ".join(f"{k}={v}" for k, v in sorted(versions.items()))
        print(f"  library_versions: {joined}")


def summarize_coverage(episodes: list[EpisodeRecord]) -> None:
    fruits = Counter(ep.fruit_type.value for ep in episodes)
    weathers = Counter(ep.ambient_weather.value for ep in episodes)
    sources = Counter(ep.source_node for ep in episodes)
    targets = Counter(ep.target_node for ep in episodes)
    print("\n[coverage]")
    print(f"  fruits: {dict(fruits)}")
    print(f"  weathers: {dict(weathers)}")
    print(f"  sources: {dict(sources)}")
    print(f"  targets: {dict(targets)}")


def summarize_labels(episodes: list[EpisodeRecord]) -> None:
    total = len(episodes)
    labels = Counter(ep.final_label for ep in episodes)
    print("\n[labels overall]")
    print(f"  fresh(0): {_fmt_pct(labels[0], total)}   spoiled(1): {_fmt_pct(labels[1], total)}")

    by_fruit: dict[str, Counter] = defaultdict(Counter)
    for ep in episodes:
        by_fruit[ep.fruit_type.value][ep.final_label] += 1
    print("[labels by fruit]")
    for fruit in sorted(by_fruit):
        cnt = by_fruit[fruit]
        n = sum(cnt.values())
        print(f"  {fruit:<10} fresh={_fmt_pct(cnt[0], n)}  spoiled={_fmt_pct(cnt[1], n)}")

    by_weather: dict[str, Counter] = defaultdict(Counter)
    for ep in episodes:
        by_weather[ep.ambient_weather.value][ep.final_label] += 1
    print("[labels by weather]")
    for w in sorted(by_weather):
        cnt = by_weather[w]
        n = sum(cnt.values())
        print(f"  {w:<8} fresh={_fmt_pct(cnt[0], n)}  spoiled={_fmt_pct(cnt[1], n)}")


def summarize_delivery(episodes: list[EpisodeRecord]) -> None:
    total = len(episodes)
    delivered = sum(1 for ep in episodes if ep.steps and ep.steps[-1].delivered)
    print("\n[delivery]")
    print(f"  delivered: {_fmt_pct(delivered, total)}")

    by_fruit: dict[str, list[int]] = defaultdict(list)
    for ep in episodes:
        flag = 1 if ep.steps and ep.steps[-1].delivered else 0
        by_fruit[ep.fruit_type.value].append(flag)
    print("[delivery by fruit]")
    for fruit in sorted(by_fruit):
        vals = by_fruit[fruit]
        print(f"  {fruit:<10} rate={sum(vals) / len(vals):.1%}  n={len(vals)}")


def summarize_lengths(episodes: list[EpisodeRecord]) -> None:
    lengths = [len(ep.steps) for ep in episodes]
    ambient = [ep.ambient_temp_c for ep in episodes]
    print("\n[episode length]")
    print(f"  min={min(lengths)}  max={max(lengths)}  mean={mean(lengths):.2f}")
    print("[ambient_temp_c]")
    print(f"  min={min(ambient):.2f}  max={max(ambient):.2f}  mean={mean(ambient):.2f}")


def summarize_risk_trajectory(episodes: list[EpisodeRecord], bins: int = 10) -> None:
    bin_values: list[list[float]] = [[] for _ in range(bins)]
    for ep in episodes:
        n = len(ep.steps)
        if n == 0:
            continue
        for i, step in enumerate(ep.steps):
            idx = min(bins - 1, i * bins // n)
            bin_values[idx].append(step.spoilage_risk)
    print(f"\n[spoilage_risk over episode progress ({bins} bins)]")
    print(f"  {'bin':>4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'n':>8}")
    for i, vals in enumerate(bin_values):
        if not vals:
            continue
        arr = np.asarray(vals)
        print(
            f"  {i:>4} {arr.mean():>8.3f} {arr.std():>8.3f} "
            f"{arr.min():>8.3f} {arr.max():>8.3f} {len(vals):>8}"
        )


def summarize_disruptions(episodes: list[EpisodeRecord]) -> None:
    per_step_counts: Counter = Counter()
    total_steps = 0
    episodes_with_any = 0
    for ep in episodes:
        seen = False
        for step in ep.steps:
            total_steps += 1
            for d in step.active_disruptions:
                per_step_counts[d.type.value] += 1
                seen = True
        if seen:
            episodes_with_any += 1
    print("\n[disruptions]")
    print(f"  total_active_across_steps: {sum(per_step_counts.values())} over {total_steps} steps")
    print(f"  episodes with ≥1 active disruption: {_fmt_pct(episodes_with_any, len(episodes))}")
    for k in sorted(per_step_counts):
        print(f"  {k}: {per_step_counts[k]}")


def summarize_actions(episodes: list[EpisodeRecord]) -> None:
    routing: Counter = Counter()
    delivery: Counter = Counter()
    spoilage: Counter = Counter()
    temperatures: list[float] = []
    inventory: list[float] = []
    for ep in episodes:
        for step in ep.steps:
            a = step.actions
            routing[int(a["routing"])] += 1
            delivery[int(a["delivery"])] += 1
            spoilage[int(a["spoilage"])] += 1
            temperatures.append(float(np.asarray(a["temperature"]).flatten()[0]))
            inventory.append(float(np.asarray(a["inventory"]).flatten()[0]))

    def _print_discrete(name: str, counter: Counter) -> None:
        total = sum(counter.values())
        print(f"[actions {name}]")
        for k in sorted(counter):
            print(f"  {k}: {_fmt_pct(counter[k], total)}")

    print("\n[actions]")
    _print_discrete("routing", routing)
    _print_discrete("delivery", delivery)
    _print_discrete("spoilage_pred", spoilage)
    t_arr = np.asarray(temperatures)
    i_arr = np.asarray(inventory)
    print(
        f"[actions temperature (°C)] mean={t_arr.mean():.2f} std={t_arr.std():.2f} "
        f"min={t_arr.min():.2f} max={t_arr.max():.2f}"
    )
    print(
        f"[actions inventory_delta] mean={i_arr.mean():.3f} std={i_arr.std():.3f} "
        f"min={i_arr.min():.3f} max={i_arr.max():.3f}"
    )


def summarize_sensor_temp(episodes: list[EpisodeRecord]) -> None:
    by_fruit: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        for step in ep.steps:
            by_fruit[ep.fruit_type.value].append(step.sensor_temperature_c)
    print("\n[sensor_temperature_c by fruit]")
    for fruit in sorted(by_fruit):
        arr = np.asarray(by_fruit[fruit])
        print(
            f"  {fruit:<10} mean={arr.mean():>7.2f} std={arr.std():>6.2f} "
            f"min={arr.min():>7.2f} max={arr.max():>7.2f}"
        )


def summarize_energy(episodes: list[EpisodeRecord]) -> None:
    values: list[float] = []
    for ep in episodes:
        for step in ep.steps:
            values.append(step.energy_usage)
    if not values:
        return
    arr = np.asarray(values)
    print("\n[energy_usage]")
    print(
        f"  mean={arr.mean():.3f} std={arr.std():.3f} "
        f"min={arr.min():.3f} max={arr.max():.3f} total_steps={len(values)}"
    )


def summarize_obs_health(episodes: list[EpisodeRecord]) -> None:
    nan_counts: Counter = Counter()
    inf_counts: Counter = Counter()
    total_per_agent: Counter = Counter()
    for ep in episodes:
        for step in ep.steps:
            for agent, arr in step.observations.items():
                total_per_agent[agent] += 1
                if np.isnan(arr).any():
                    nan_counts[agent] += 1
                if np.isinf(arr).any():
                    inf_counts[agent] += 1
    print("\n[obs health]")
    for agent in sorted(total_per_agent):
        print(
            f"  {agent:<12} steps={total_per_agent[agent]:>6} "
            f"nan_steps={nan_counts[agent]} inf_steps={inf_counts[agent]}"
        )


def inspect(dataset_dir: Path) -> None:
    summarize_manifest(dataset_dir)
    episodes = load_episodes(dataset_dir)
    if not episodes:
        print(f"No episodes found in {dataset_dir}")
        return
    print(f"\nLoaded {len(episodes)} episodes from {dataset_dir}")
    summarize_coverage(episodes)
    summarize_labels(episodes)
    summarize_delivery(episodes)
    summarize_lengths(episodes)
    summarize_risk_trajectory(episodes)
    summarize_disruptions(episodes)
    summarize_actions(episodes)
    summarize_sensor_temp(episodes)
    summarize_energy(episodes)
    summarize_obs_health(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a coldchain-marl synthetic dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.dataset)


if __name__ == "__main__":
    main()

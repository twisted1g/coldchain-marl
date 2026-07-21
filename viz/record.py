"""Roll out a rolling inference episode and serialize it to JSONL.

Output is a JSONL stream: the first line is a ``meta`` record (graph layout,
fruit thresholds, horizon), each following line is a ``tick`` record (shipment,
inventory, vehicles, disruptions, rewards, negotiations). The dashboard renderer
consumes this file; the same records stream live over SSE from ``viz.server``.

The rollout itself lives in ``viz.inference`` (shared with the live stream); this
module just collects the stream into a file and provides the CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.config import ARTIFACTS
from viz.inference import DEFAULT_HORIZON, run_inference


def record_episode(
    seed: int,
    tag: str | None,
    scenario_id: str | None,
    max_steps: int | None,
    mediator: str | None = None,
) -> list[dict[str, Any]]:
    """Collect a rolling inference episode. ``max_steps`` sets the rolling
    horizon (ticks); ``None`` uses the default."""
    horizon = max_steps if max_steps is not None else DEFAULT_HORIZON
    return list(
        run_inference(
            seed,
            tag,
            mediator=mediator,
            horizon=horizon,
            scenario_id=scenario_id,
        )
    )


def write_episode(records: list[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return sum(1 for r in records if r["type"] == "tick")


def episode_name(seed: int, tag: str | None) -> str:
    return f"episode_{seed}" + (f"_{tag}" if tag else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--episodes", type=int, default=1,
                        help="record N consecutive seeds (seed, seed+1, ...)")
    parser.add_argument("--tag", default=None, help="module variant, e.g. scn05")
    parser.add_argument("--scenario", default=None, help="LLM scenario id (needs bank)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="rolling horizon in ticks (default: %d)" % DEFAULT_HORIZON)
    parser.add_argument(
        "--mediator",
        default="off",
        choices=["off", "greedy", "llm"],
        help="resolve delivery-slot conflicts (Alg 6) before each step",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL (single episode only; default artifacts/episodes/<name>.jsonl)",
    )
    args = parser.parse_args()

    episodes_dir = ARTIFACTS / "episodes"
    for k in range(args.episodes):
        seed = args.seed + k
        records = record_episode(
            seed, args.tag, args.scenario, args.max_steps, args.mediator
        )
        out = (
            args.out
            if args.out is not None and args.episodes == 1
            else episodes_dir / f"{episode_name(seed, args.tag)}.jsonl"
        )
        ticks = write_episode(records, out)
        print(f"wrote {ticks} ticks -> {out}")


if __name__ == "__main__":
    main()

"""Continuous live inference for the dashboard SSE stream.

Thin adapter over ``viz.inference.run_inference`` (shared with the offline
recorder). Kept as its own module so ``viz.server`` imports the live entrypoint
without pulling the recorder's file-writing helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from viz.inference import DEFAULT_HORIZON, DEFAULT_SLOT_SPAN, run_inference

__all__ = ["DEFAULT_HORIZON", "live_stream"]


def live_stream(
    seed: int,
    tag: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    max_steps: int | None = None,
    mediator: str | None = "llm",
) -> Iterator[dict[str, Any]]:
    """Rolling multi-shipment inference streamed for the live dashboard.

    ``horizon`` is the number of ticks to stream; ``max_steps`` sizes the
    delivery slot windows (dense dispatch when short); ``mediator``
    ("off"/"greedy"/"llm") resolves slot conflicts (Alg 6) before each step.
    """
    yield from run_inference(
        seed,
        tag,
        mediator=mediator,
        horizon=horizon,
        slot_span=max_steps or DEFAULT_SLOT_SPAN,
    )

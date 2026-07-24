"""Phase V visualization: capture episodes and render the world dashboard.

``record`` rolls out one episode with trained policies and serializes the
per-tick world state to JSONL; ``dashboard`` reads that JSONL and renders the
frames. Capture and render are decoupled so later panels (V2 agents, V3
negotiation, V5 ledger) and alternative renderers can subscribe to the same
event stream without touching the rollout.
"""

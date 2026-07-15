"""Generate the offline disruption-scenario bank via a local LLM.

Usage:
    uv run python -m llm.generate_bank --model nvidia/nemotron-3-nano-4b -n 10

Produces data/scenarios/bank.json: validated, deduplicated scenarios per
category plus a manifest (model, endpoint, timestamp, reject stats).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from llm.client import LLMConfig, LLMError, OpenAICompatClient
from llm.scenarios import (
    CATEGORY_EFFECTS,
    DURATION_RANGE,
    INTEGER_KINDS,
    LLM_RESPONSE_SCHEMA,
    MAGNITUDE_RANGES,
    TARGETED_KINDS,
    Scenario,
    ScenarioCategory,
    ScenarioValidationError,
    parse_scenario,
    save_bank,
)

DEFAULT_OUT = Path("data/scenarios/bank.json")
MAX_ATTEMPTS_PER_SCENARIO = 5

SYSTEM_PROMPT = """\
You generate disruption scenarios for a fruit cold-chain logistics simulation.
The network has farms, hubs, distribution centers (dc), and retailers.
Each scenario must be plausible, specific, and internally consistent:
severity, duration, and effect magnitudes must agree with the description.
Respond with JSON only."""


def _effect_spec(category: ScenarioCategory) -> str:
    lines = []
    for kind in sorted(CATEGORY_EFFECTS[category]):
        low, high = MAGNITUDE_RANGES[kind]
        numtype = "integer" if kind in INTEGER_KINDS else "float"
        target = (
            "target_role: farm|hub|dc|retailer|any"
            if kind in TARGETED_KINDS
            else 'target_role: must be "any"'
        )
        lines.append(f'- kind "{kind}": magnitude {numtype} in [{low}, {high}]; {target}')
    return "\n".join(lines)


def _user_prompt(category: ScenarioCategory, variant: int) -> str:
    return f"""\
Generate ONE disruption scenario of category "{category}".
This is variant {variant}; make it distinct from typical examples.

Constraints (violations are rejected):
- severity: float in [0.0, 1.0], reflecting overall impact
- duration_steps: integer in [{DURATION_RANGE[0]}, {DURATION_RANGE[1]}] (1 step = 1 day)
- effects: 1 to {len(CATEGORY_EFFECTS[category])} entries, no duplicate kinds,
  only these kinds with magnitudes strictly inside the stated ranges:
{_effect_spec(category)}
- description: 1-3 sentences of concrete operational detail."""


def generate_category(
    client: OpenAICompatClient,
    category: ScenarioCategory,
    count: int,
) -> tuple[list[Scenario], int]:
    scenarios: list[Scenario] = []
    signatures = set()
    rejects = 0
    for variant in range(1, count + 1):
        scenario_id = f"{category}_{variant:03d}"
        for _ in range(MAX_ATTEMPTS_PER_SCENARIO):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(category, variant)},
            ]
            try:
                payload = client.complete_json(
                    messages, LLM_RESPONSE_SCHEMA, schema_name="disruption_scenario"
                )
                scenario = parse_scenario(payload, scenario_id, category)
            except (LLMError, ScenarioValidationError) as exc:
                rejects += 1
                print(f"  reject {scenario_id}: {exc}", file=sys.stderr)
                continue
            if scenario.signature() in signatures:
                rejects += 1
                print(f"  reject {scenario_id}: duplicate", file=sys.stderr)
                continue
            signatures.add(scenario.signature())
            scenarios.append(scenario)
            break
        else:
            print(f"  gave up on {scenario_id}", file=sys.stderr)
    return scenarios, rejects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("-n", "--per-category", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    overrides = {"model": args.model, "temperature": args.temperature}
    if args.base_url:
        overrides["base_url"] = args.base_url
    config = LLMConfig.from_env(**overrides)

    all_scenarios: list[Scenario] = []
    total_rejects = 0
    with OpenAICompatClient(config) as client:
        for category in ScenarioCategory:
            print(f"{category}: generating {args.per_category}...")
            scenarios, rejects = generate_category(client, category, args.per_category)
            print(f"{category}: kept {len(scenarios)}, rejected {rejects}")
            all_scenarios.extend(scenarios)
            total_rejects += rejects

    manifest = {
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "per_category_requested": args.per_category,
        "total_kept": len(all_scenarios),
        "total_rejected": total_rejects,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_bank(all_scenarios, str(args.out), manifest)
    print(f"wrote {len(all_scenarios)} scenarios to {args.out}")


if __name__ == "__main__":
    main()

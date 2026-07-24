import type { ReactNode } from "react";
import type { Tick } from "../types";

interface Props {
  tick: Tick | null;
}

/** Sum a metric across a group's info dicts (e.g. every inventory_* agent). */
function sumInfo(
  infos: Record<string, Record<string, number>>,
  prefix: string,
  key: string,
): number {
  let total = 0;
  for (const [name, info] of Object.entries(infos)) {
    if (name === prefix || name.startsWith(`${prefix}_`)) total += info[key] ?? 0;
  }
  return total;
}

/** Sum the live reward of every policy in a group. */
function sumReward(rewards: Record<string, number>, prefix: string): number {
  let total = 0;
  for (const [name, r] of Object.entries(rewards)) {
    if (name === prefix || name.startsWith(`${prefix}_`)) total += r;
  }
  return total;
}

/** Count group instances that satisfy a predicate on their info dict. */
function countInfo(
  infos: Record<string, Record<string, number>>,
  prefix: string,
  pred: (info: Record<string, number>) => boolean,
): number {
  let n = 0;
  for (const [name, info] of Object.entries(infos)) {
    if (name.startsWith(`${prefix}_`) && pred(info)) n++;
  }
  return n;
}

function Metric({ k, v, bad }: { k: string; v: ReactNode; bad?: boolean }) {
  return (
    <div className="ac-row">
      <span>{k}</span>
      <b className={bad ? "warnval" : undefined}>{v}</b>
    </div>
  );
}

interface Card {
  name: string;
  accent: string;
  count: number;
  reward: number;
  rows: { k: string; v: ReactNode; bad?: boolean }[];
}

/**
 * Bottom strip of MARL agent status cards — one per logical agent group. Each
 * card shows the group's live summed reward, its instance count, and the two or
 * three headline metrics from that agent's `info` dict. Collapsing the four
 * per-vehicle routing/temperature/spoilage, four inventory and three delivery
 * instances into one card each keeps it reading as the paper's five cooperating
 * agents rather than sixteen scattered policies.
 */
export function AgentsPanel({ tick }: Props) {
  if (!tick) return null;
  const { rewards, infos } = tick;

  // routing / temperature / spoilage are now one policy instance per vehicle
  // (singleton eliminated). Idle trucks report zeroed info, so the per-crate
  // metrics average over the trucks actually carrying a crate.
  const nVeh = tick.vehicles.length;
  const active = Math.max(1, tick.vehicles.filter((v) => v.crate != null).length);

  const cards: Card[] = [
    {
      name: "temperature",
      accent: "#0891b2",
      count: nVeh,
      reward: sumReward(rewards, "temperature"),
      rows: [
        {
          k: "Δ temp",
          v: `${(sumInfo(infos, "temperature", "temp_deviation") / active).toFixed(3)}`,
        },
      ],
    },
    {
      name: "routing",
      accent: "#2563eb",
      count: nVeh,
      reward: sumReward(rewards, "routing"),
      rows: [
        {
          k: "route cost",
          v: `${sumInfo(infos, "routing", "route_cost").toFixed(2)}`,
        },
        { k: "delivered", v: `${sumInfo(infos, "routing", "delivered").toFixed(0)}` },
      ],
    },
    {
      name: "spoilage",
      accent: "#16a34a",
      count: nVeh,
      reward: sumReward(rewards, "spoilage"),
      rows: [
        {
          k: "predict",
          v: `${(sumInfo(infos, "spoilage", "y_pred") / active).toFixed(2)}`,
        },
        {
          k: "FN rate",
          v: `${sumInfo(infos, "spoilage", "fn_rate").toFixed(2)}`,
          bad: sumInfo(infos, "spoilage", "fn_rate") > 0.01,
        },
      ],
    },
    {
      name: "inventory",
      accent: "#d97706",
      count: tick.inventory.levels.length,
      reward: sumReward(rewards, "inventory"),
      rows: [
        {
          k: "unmet demand",
          v: sumInfo(infos, "inventory", "unmet_demand").toFixed(2),
          bad: sumInfo(infos, "inventory", "unmet_demand") > 0.01,
        },
        {
          k: "ordering",
          v: `${countInfo(infos, "inventory", (i) => (i.order ?? 0) > 0.01)}/${
            tick.inventory.levels.length
          }`,
        },
      ],
    },
    {
      name: "delivery",
      accent: "#db2777",
      count: tick.vehicles.length,
      reward: sumReward(rewards, "delivery"),
      rows: [
        {
          k: "SLA breaches",
          v: countInfo(infos, "delivery", (i) => (i.sla_violated ?? 0) > 0.5),
          bad: countInfo(infos, "delivery", (i) => (i.sla_violated ?? 0) > 0.5) > 0,
        },
        {
          k: "conflicts",
          v: countInfo(infos, "delivery", (i) => (i.conflict ?? 0) > 0.5),
          bad: countInfo(infos, "delivery", (i) => (i.conflict ?? 0) > 0.5) > 0,
        },
      ],
    },
  ];

  return (
    <section className="card area-agents">
      <div className="cardhead">
        <h2>MARL agents</h2>
        <span className="hint">live reward · per-agent telemetry</span>
      </div>
      <div className="agents">
        {cards.map((c) => (
          <div
            key={c.name}
            className="agentcard"
            style={{ borderTopColor: c.accent }}
          >
            <div className="ac-head">
              <span className="ac-name">{c.name}</span>
              {c.count > 1 && <span className="badge">×{c.count}</span>}
            </div>
            <div className="ac-body">
              <div className="ac-row">
                <span>reward</span>
                <b>
                  {c.reward >= 0 ? "+" : ""}
                  {c.reward.toFixed(2)}
                </b>
              </div>
              {c.rows.map((r) => (
                <Metric key={r.k} k={r.k} v={r.v} bad={r.bad} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-dist-min";
import { Plot } from "../Plot";
import { nodePositions } from "../graphLayout";
import type { Meta, Tick } from "../types";
import { KIND_COLOR } from "../types";

export type HoverTarget =
  | { t: "node"; name: string }
  | { t: "veh"; i: number };

interface Props {
  meta: Meta;
  tick: Tick | null;
  onHover: (target: HoverTarget, x: number, y: number) => void;
  onUnhover: () => void;
}

const GRID = "#cbd5e1";
const INK = "#1e293b";

/** One distinct hue per delivery agent / vehicle, cycled if the fleet is large. */
const VEHICLE_COLOR = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ca8a04"];
const vehColor = (i: number) => VEHICLE_COLOR[i % VEHICLE_COLOR.length];

/** Green (fresh) -> red (spoiled) along a spoilage risk in [0,1]. */
function riskColor(r: number): string {
  const hue = (1 - Math.max(0, Math.min(1, r))) * 120;
  return `hsl(${hue}, 65%, 48%)`;
}

/**
 * System topology: farm -> hub -> DC -> retail laid out in columns, edges as the
 * physical links, the live cold-chain shipment as a risk-colored diamond, and a
 * ring around its target retailer. Node overlays (fleet, cargo) live elsewhere.
 */
export function GraphView({ meta, tick, onHover, onUnhover }: Props) {
  const pos = useMemo(() => nodePositions(meta), [meta]);

  const data = useMemo<Data[]>(() => {
    const names = meta.nodes.map((n) => n.name);
    const kindOf: Record<string, string> = {};
    for (const n of meta.nodes) kindOf[n.name] = n.kind;

    // edges as one null-separated line trace
    const ex: (number | null)[] = [];
    const ey: (number | null)[] = [];
    for (const [u, v] of meta.edges) {
      if (!pos[u] || !pos[v]) continue;
      ex.push(pos[u].x, pos[v].x, null);
      ey.push(pos[u].y, pos[v].y, null);
    }

    const traces: Data[] = [
      {
        x: ex,
        y: ey,
        mode: "lines",
        type: "scatter",
        hoverinfo: "skip",
        line: { color: GRID, width: 1.4 },
      },
      {
        x: names.map((n) => pos[n].x),
        y: names.map((n) => pos[n].y),
        mode: "text+markers",
        type: "scatter",
        text: names.map((n) => n.replace("_", " ")),
        textposition: "bottom center",
        textfont: { color: "#475569", size: 9 },
        hoverinfo: "none",
        // Plotly types customdata as primitives; objects pass through untouched.
        customdata: names.map((n) => ({ t: "node", name: n })) as unknown as number[],
        marker: {
          size: 26,
          color: names.map((n) => KIND_COLOR[kindOf[n] as keyof typeof KIND_COLOR] ?? INK),
          line: { width: 2, color: "#ffffff" },
        },
      },
    ];

    // One colored ring per delivery agent around the retailer it serves, plus a
    // matching truck marker at the vehicle's live position (fanned when several
    // trucks sit on the same node).
    const vehicles = tick?.vehicles ?? [];
    const atNode: Record<string, number[]> = {};
    vehicles.forEach((v, i) => (atNode[v.current_node] ??= []).push(i));

    // Draw each loaded truck's remaining road to its destination retailer so the
    // journey is legible (the truck's shortest path = meta.restock_paths[dest],
    // sliced from where it currently sits). Colored by agent, drawn under the
    // markers.
    vehicles.forEach((v, i) => {
      if (v.carrying == null) return;
      const full = meta.restock_paths?.[v.carrying];
      if (!full || !full.length) return;
      const at = full.indexOf(v.current_node);
      const remaining = at >= 0 ? full.slice(at) : full;
      const lx: (number | null)[] = [];
      const ly: (number | null)[] = [];
      for (let k = 0; k < remaining.length - 1; k++) {
        const a = pos[remaining[k]];
        const b = pos[remaining[k + 1]];
        if (!a || !b) continue;
        lx.push(a.x, b.x, null);
        ly.push(a.y, b.y, null);
      }
      if (!lx.length) return;
      traces.push({
        x: lx,
        y: ly,
        mode: "lines",
        type: "scatter",
        hoverinfo: "skip",
        line: { color: vehColor(i), width: 2.5 },
        opacity: 0.5,
      });
    });

    // Ring the retailer each loaded truck is actually delivering to (the order's
    // retailer = retail_{carrying}), in that agent's color. Idle trucks carry no
    // order, so no ring — the destination moves trip to trip.
    vehicles.forEach((v, i) => {
      if (v.carrying == null) return;
      const p = pos[`retail_${v.carrying}`];
      if (!p) return;
      traces.push({
        x: [p.x],
        y: [p.y],
        mode: "markers",
        type: "scatter",
        hoverinfo: "skip",
        marker: {
          size: 42,
          color: "rgba(0,0,0,0)",
          line: { width: 2.5, color: vehColor(i) },
        },
      });
    });

    vehicles.forEach((v, i) => {
      const base = pos[v.current_node];
      if (!base) return;
      // Queue trucks sharing a node single-file along the travel axis (x), each
      // trailing behind the one ahead so the backlog reads left-to-right.
      const group = atNode[v.current_node] ?? [i];
      const k = group.indexOf(i);
      const p = { x: base.x - k * 0.055, y: base.y };
      traces.push({
        x: [p.x],
        y: [p.y],
        mode: "markers",
        type: "scatter",
        hoverinfo: "none",
        customdata: [{ t: "veh", i }] as unknown as number[],
        marker: {
          size: 15,
          symbol: "triangle-up",
          color: vehColor(i),
          // outline reddens with the crate's own spoilage risk (white if empty)
          line: {
            width: v.crate ? 2.5 : 1.5,
            color: v.crate ? riskColor(v.crate.spoilage_risk) : "#ffffff",
          },
        },
      });
    });

    return traces;
  }, [meta, pos, tick]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { l: 8, r: 8, t: 8, b: 8 },
      hovermode: "closest",
      // left pad leaves room for trucks queued behind the source depot (x=0)
      xaxis: { visible: false, range: [-0.28, 1.1], fixedrange: true },
      // flip Y so the first row sits at the top, reading top-to-bottom
      yaxis: { visible: false, range: [1.12, -0.12], fixedrange: true },
    }),
    [],
  );

  return (
    <section className="card area-graph">
      <div className="cardhead">
        <h2>System graph</h2>
        <span className="hint">
          {meta.source} → {meta.target} · {meta.nodes.length} nodes
        </span>
      </div>
      <Plot
        className="graph"
        data={data}
        layout={layout}
        onPointHover={(cd, x, y) => onHover(cd as HoverTarget, x, y)}
        onPointUnhover={onUnhover}
      />
    </section>
  );
}

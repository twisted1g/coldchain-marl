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

/** Green (fresh) -> red (spoiled) along a spoilage risk in [0,1]. Vivid and fully
 *  opaque so the crate's condition reads at a glance. */
function riskColor(r: number): string {
  const hue = (1 - Math.max(0, Math.min(1, r))) * 120;
  return `hsl(${hue}, 90%, 45%)`;
}

/**
 * System topology: farm -> hub -> DC -> retail laid out in columns, edges as the
 * physical links. Each delivery truck is a triangle in its agent's hue; a loaded
 * truck carries a square crate badge coloured by spoilage risk, and its live
 * destination retailer wears a matching ring. No route line is drawn — the
 * dynamic-assignment target can change trip to trip, so a fixed path misleads.
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
      const loaded = v.crate != null;
      // Truck glyph: filled triangle in the agent's hue, fully opaque. Empty trucks
      // are smaller with a thin grey outline; loaded trucks are larger so cargo
      // motion stands out.
      traces.push({
        x: [p.x],
        y: [p.y],
        mode: "markers",
        type: "scatter",
        hoverinfo: "none",
        customdata: [{ t: "veh", i }] as unknown as number[],
        marker: {
          size: loaded ? 18 : 12,
          symbol: "triangle-up",
          color: vehColor(i),
          line: { width: 1.5, color: loaded ? "#ffffff" : "#94a3b8" },
        },
      });
      // Crate badge: an amber box floating above a loaded truck. Amber is a fixed
      // cargo colour that never matches a node hue (green/blue/amber-dc/pink), so a
      // crate is always legible against the network. Its spoilage risk shows as the
      // frame colour (green fresh -> red spoiled). Shape + colour make the cargo —
      // and its condition — the most distinct thing moving.
      if (loaded && v.crate) {
        traces.push({
          x: [p.x],
          y: [p.y - 0.06],
          mode: "markers",
          type: "scatter",
          hoverinfo: "none",
          customdata: [{ t: "veh", i }] as unknown as number[],
          marker: {
            size: 16,
            symbol: "square",
            color: "#f59e0b",
            line: { width: 3.5, color: riskColor(v.crate.spoilage_risk) },
          },
        });
      }
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

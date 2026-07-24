import type { CSSProperties, ReactNode } from "react";
import type { Meta, Tick } from "../types";
import { KIND_LABEL } from "../types";
import type { HoverTarget } from "./GraphView";

interface Props {
  target: HoverTarget | null;
  pos: { x: number; y: number } | null;
  meta: Meta;
  tick: Tick | null;
}

const VEHICLE_COLOR = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ca8a04"];
const vehColor = (i: number) => VEHICLE_COLOR[i % VEHICLE_COLOR.length];

/** trailing integer of e.g. "retail_2" -> 2, or null. */
function instanceOf(name: string): number | null {
  const m = /_(\d+)$/.exec(name);
  return m ? Number(m[1]) : null;
}

function Line({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="hc-row">
      <span className="hc-k">{k}</span>
      <span className="hc-v">{v}</span>
    </div>
  );
}

/**
 * Small floating readout pinned next to the hovered graph element. Trucks show
 * their delivery agent's live params (compact); nodes show topology plus any
 * inventory / shipment state sitting on that node.
 */
export function HoverCard({ target, pos, meta, tick }: Props) {
  if (!target || !pos) return null;

  const style: CSSProperties = { left: pos.x + 14, top: pos.y + 14 };

  if (target.t === "veh") {
    const v = tick?.vehicles[target.i];
    if (!v) return null;
    const color = vehColor(target.i);
    const status =
      v.carrying == null
        ? "idle"
        : v.current_node === meta.source
          ? "scheduled"
          : "in transit";
    const reward = tick?.rewards?.[`delivery_${target.i}`];
    return (
      <div className="hovercard" style={style}>
        <div className="hc-head">
          <span className="swatch" style={{ background: color }} />
          delivery_{target.i}
          <span className="hc-badge">{status}</span>
        </div>
        <Line
          k="delivering to"
          v={v.carrying == null ? "—" : `retail_${v.carrying}`}
        />
        <Line k="at" v={v.current_node.replace("_", " ")} />
        <Line k="slot" v={v.chosen_slot} />
        <Line k="delay" v={`${v.delay.toFixed(2)}t`} />
        {v.crate && (
          <>
            <Line k="crate temp" v={`${v.crate.sensor_temp.toFixed(1)}°C`} />
            <Line k="setpoint" v={`${v.crate.desired_temp.toFixed(1)}°C`} />
            <Line
              k="spoilage"
              v={
                v.crate.spoilage_risk > 0.4 ? (
                  <span className="tag bad">{v.crate.spoilage_risk.toFixed(2)}</span>
                ) : (
                  v.crate.spoilage_risk.toFixed(2)
                )
              }
            />
            <Line k="freshness" v={v.crate.freshness_score.toFixed(2)} />
            {v.crate.spoilage_prediction != null && (
              <Line
                k="predicted"
                v={v.crate.spoilage_prediction.toFixed(2)}
              />
            )}
          </>
        )}
        {(v.conflict || v.sla_violated) && (
          <Line
            k="flags"
            v={
              <>
                {v.conflict && <span className="tag bad">conflict</span>}
                {v.sla_violated && <span className="tag bad">SLA✗</span>}
              </>
            }
          />
        )}
        {reward != null && <Line k="reward" v={reward.toFixed(3)} />}
      </div>
    );
  }

  // node
  const name = target.name;
  const node = meta.nodes.find((n) => n.name === name);
  const kind = node?.kind;
  const inst = instanceOf(name);
  const inv = tick?.inventory;
  const showInv =
    kind === "retail" && inv && inst != null && inst < inv.levels.length;

  // live per-node micro-climate (Design F) + its kind setpoint/band
  const climate = tick?.node_climate?.[name];
  const band = node?.climate_band;
  const setpoint = node?.climate_setpoint;

  // Trucks parked on this node right now, with what they carry. Gives every node
  // — farm/hub/dc included — live content instead of a bare "no live state".
  const here = (tick?.vehicles ?? [])
    .map((v, i) => ({ v, i }))
    .filter(({ v }) => v.current_node === name);

  const hasState = showInv || here.length > 0 || climate != null;

  return (
    <div className="hovercard" style={style}>
      <div className="hc-head">
        {name.replace("_", " ")}
        {kind && <span className="hc-badge">{KIND_LABEL[kind]}</span>}
      </div>
      {climate && (
        <>
          <Line k="storage temp" v={`${climate.temp.toFixed(1)}°C`} />
          <Line k="humidity" v={`${(climate.humidity * 100).toFixed(0)}%`} />
          {setpoint != null && band && (
            <Line
              k="setpoint"
              v={`${setpoint.toFixed(0)}° (${band[0].toFixed(0)}–${band[1].toFixed(0)})`}
            />
          )}
        </>
      )}
      {showInv && (
        <>
          <Line k="stock" v={inv!.levels[inst!].toFixed(1)} />
          <Line k="order" v={inv!.order[inst!].toFixed(1)} />
          <Line k="demand" v={inv!.demand_today[inst!].toFixed(1)} />
          <Line
            k="unmet"
            v={
              inv!.unmet[inst!] > 0.01 ? (
                <span className="tag bad">{inv!.unmet[inst!].toFixed(2)}</span>
              ) : (
                inv!.unmet[inst!].toFixed(2)
              )
            }
          />
        </>
      )}
      {here.map(({ v, i }) => (
        <Line
          key={i}
          k={
            <>
              <span className="swatch" style={{ background: vehColor(i) }} />
              delivery_{i}
            </>
          }
          v={
            v.carrying == null ? (
              "empty"
            ) : (
              <>
                crate → retail_{v.carrying}
                {v.crate && v.crate.spoilage_risk > 0.4 && (
                  <span className="tag bad" style={{ marginLeft: 4 }}>
                    {v.crate.spoilage_risk.toFixed(2)}
                  </span>
                )}
              </>
            )
          }
        />
      ))}
      {!hasState && <div className="hc-row muted">no traffic</div>}
    </div>
  );
}

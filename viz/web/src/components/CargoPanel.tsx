import type { Tick } from "../types";

interface Props {
  tick: Tick | null;
}

const VEHICLE_COLOR = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ca8a04"];
const vehColor = (i: number) => VEHICLE_COLOR[i % VEHICLE_COLOR.length];

function riskCls(r: number): string {
  if (r > 0.4) return "bad";
  if (r > 0.15) return "warn";
  return "good";
}

/**
 * Per-crate cold-chain readout — one row per loaded truck. Each crate carries its
 * own thermal + spoilage state (the temperature policy is shared, the resulting
 * temp is not), so the table exposes sensor temp vs setpoint alongside the crate's
 * live spoilage risk and freshness. Idle trucks carry nothing and are skipped.
 */
export function CargoPanel({ tick }: Props) {
  if (!tick) return null;
  const crates = tick.vehicles
    .map((v, i) => ({ v, i }))
    .filter(({ v }) => v.crate != null);

  const worst = crates.reduce((m, { v }) => Math.max(m, v.crate!.spoilage_risk), 0);

  return (
    <section className="card area-charts">
      <div className="cardhead">
        <h2>Crate cold-chain</h2>
        <span className={`tag ${riskCls(worst)}`}>
          {crates.length ? `worst risk ${worst.toFixed(2)}` : "no cargo"}
        </span>
      </div>
      {crates.length === 0 ? (
        <p className="hint">All trucks idle — no crates in transit.</p>
      ) : (
        <table className="invtable">
          <thead>
            <tr>
              <th>crate</th>
              <th>at</th>
              <th>temp</th>
              <th>setpoint</th>
              <th>Δ</th>
              <th>spoilage</th>
              <th>pred</th>
              <th>freshness</th>
            </tr>
          </thead>
          <tbody>
            {crates.map(({ v, i }) => {
              const c = v.crate!;
              const dev = c.sensor_temp - c.desired_temp;
              return (
                <tr key={i}>
                  <td>
                    <span
                      className="swatch"
                      style={{ background: vehColor(i) }}
                    />
                    → retail_{v.carrying}
                  </td>
                  <td>{v.current_node.replace("_", " ")}</td>
                  <td>{c.sensor_temp.toFixed(1)}°</td>
                  <td>{c.desired_temp.toFixed(1)}°</td>
                  <td>{dev >= 0 ? "+" : ""}{dev.toFixed(1)}</td>
                  <td>
                    <span className={`tag ${riskCls(c.spoilage_risk)}`}>
                      {c.spoilage_risk.toFixed(2)}
                    </span>
                  </td>
                  <td>
                    {c.spoilage_prediction != null
                      ? c.spoilage_prediction.toFixed(2)
                      : "—"}
                  </td>
                  <td>{c.freshness_score.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

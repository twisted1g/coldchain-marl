import type { Tick } from "../types";

interface Props {
  tick: Tick | null;
}

/**
 * Live state of every inventory agent — one row per retail instance. Each agent
 * owns one column position in the sim's inventory arrays (levels/order/unmet/
 * demand/forecast). Rows where demand went unmet flag red.
 */
export function InventoryPanel({ tick }: Props) {
  if (!tick) return null;
  const inv = tick.inventory;
  const n = inv.levels.length;
  const idx = Array.from({ length: n }, (_, i) => i);

  const totUnmet = inv.unmet.reduce((a, b) => a + b, 0);

  return (
    <section className="card">
      <div className="cardhead">
        <h2>Inventory agents</h2>
        <span className={`tag ${totUnmet > 0.01 ? "bad" : "good"}`}>
          unmet {totUnmet.toFixed(2)}
        </span>
      </div>
      <table className="invtable">
        <thead>
          <tr>
            <th>agent</th>
            <th>level</th>
            <th>order</th>
            <th>demand</th>
            <th>forecast</th>
            <th>unmet</th>
          </tr>
        </thead>
        <tbody>
          {idx.map((i) => {
            const reward = tick.rewards?.[`inventory_${i}`];
            const short = inv.unmet[i] > 0.01;
            return (
              <tr key={i} className={short ? "short" : ""} title={reward != null ? `reward ${reward.toFixed(3)}` : undefined}>
                <td>inventory_{i}</td>
                <td>{inv.levels[i].toFixed(1)}</td>
                <td>{inv.order[i].toFixed(1)}</td>
                <td>{inv.demand_today[i].toFixed(1)}</td>
                <td>{inv.forecast[i].toFixed(1)}</td>
                <td>{inv.unmet[i].toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

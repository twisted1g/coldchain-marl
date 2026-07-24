import type { Tick } from "../types";

interface Props {
  tick: Tick | null;
}

/**
 * Slot-conflict mediation (paper Alg 6): when two delivery agents pick the same
 * departure slot the mediator ("off"/"greedy"/"llm") reassigns them. Each tick's
 * ``negotiations`` carries the resolved events; ``vehicle.conflict`` flags any
 * still-contended trucks.
 */
export function NegotiationPanel({ tick }: Props) {
  if (!tick) return null;
  const events = tick.negotiations ?? [];
  const contended = tick.vehicles
    .map((v, i) => ({ v, i }))
    .filter(({ v }) => v.conflict);

  return (
    <section className="card">
      <div className="cardhead">
        <h2>Slot conflicts</h2>
        <span className={`tag ${contended.length ? "bad" : "good"}`}>
          {contended.length ? `${contended.length} contended` : "clear"}
        </span>
      </div>

      {events.length === 0 && contended.length === 0 && (
        <p className="hint">No slot contention this tick.</p>
      )}

      {contended.length > 0 && (
        <div className="queue">
          {contended.map(({ i, v }) => (
            <span key={i} className="chip">
              delivery_{i} · slot {v.chosen_slot}
            </span>
          ))}
        </div>
      )}

      <div className="negotiation">
        {events.map((e, k) => (
          <div key={k} className={`nevent ${e.resolved ? "resolved" : "failed"}`}>
            <div className="nrow">
              <span className="badge">slot {e.slot}</span>
              {e.agents.map((a) => (
                <span key={a} className="tag muted">{a}</span>
              ))}
              <span className={`tag ${e.resolved ? "good" : "bad"}`}>
                {e.resolved ? "resolved" : "failed"}
              </span>
              <span className="hint">
                {e.mediator} · {e.rounds} round{e.rounds === 1 ? "" : "s"}
              </span>
            </div>
            <div className="nslots">
              {e.agents.map((a) => (
                <span key={a} className="nslot">
                  {a}: {e.initial[a]} → <b>{e.final[a]}</b>
                </span>
              ))}
            </div>
            {e.summary && <div className="nsummary">{e.summary}</div>}
          </div>
        ))}
      </div>
    </section>
  );
}

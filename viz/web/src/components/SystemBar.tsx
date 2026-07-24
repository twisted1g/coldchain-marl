import type { Tick } from "../types";

interface Props {
  tick: Tick | null;
}

const WEATHER_ICON: Record<string, string> = {
  sunny: "☀",
  clear: "☀",
  clouds: "☁",
  cloudy: "☁",
  rain: "🌧",
  storm: "⛈",
  snow: "❄",
  fog: "🌫",
  heat: "🔥",
};

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function Tile({ label, val, sub, cls }: { label: string; val: string; sub?: string; cls?: string }) {
  return (
    <div className={`kpi ${cls ?? ""}`}>
      <span className="kpi-val">{val}</span>
      <span className="kpi-label">{label}</span>
      {sub && <span className="hint">{sub}</span>}
    </div>
  );
}

/** System-wide conditions above the graph: weather, ambient temp/humidity,
 *  calendar, and any active disruptions. */
export function SystemBar({ tick }: Props) {
  if (!tick) return null;
  const a = tick.ambient;
  const c = tick.calendar;
  const disr = tick.disruptions;
  const icon = WEATHER_ICON[a.weather.toLowerCase()] ?? "";

  return (
    <div className="kpibar">
      <Tile label="weather" val={`${icon} ${a.weather}`} />
      <Tile label="ambient temp" val={`${a.temp.toFixed(1)}°C`} />
      <Tile label="humidity" val={`${(a.humidity * 100).toFixed(0)}%`} />
      <Tile
        label="calendar"
        val={`${WEEKDAYS[c.weekday] ?? c.weekday} · d${c.day_of_year}`}
        sub={`event ×${c.event_multiplier.toFixed(2)}`}
      />
      <Tile
        label="disruptions"
        val={disr.length ? String(disr.length) : "none"}
        sub={disr.length ? disr.map((d) => `${d.type}@${d.target}`).join(", ") : undefined}
        cls={disr.length ? "bad" : "good"}
      />
      <Tile label="tick" val={String(tick.tick)} />
    </div>
  );
}

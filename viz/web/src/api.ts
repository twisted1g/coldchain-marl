import type { Episode, Meta, Tick } from "./types";

// The inference API is a separate service (viz/api.py). Base URL is injected by
// the static server as window.API_BASE; falls back to :8001 for `vite dev`.
export const API_BASE: string =
  window.API_BASE ?? `http://${location.hostname}:8001`;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export function listEpisodes(): Promise<{ episodes: string[] }> {
  return getJSON("/api/episodes");
}

export function loadEpisode(name: string): Promise<Episode> {
  return getJSON(`/api/episode/${encodeURIComponent(name)}`);
}

export interface RunRequest {
  seed?: number;
  episodes?: number;
  tag?: string | null;
  scenario?: string | null;
  max_steps?: number | null;
  mediator?: string;
}

export async function runEpisode(req: RunRequest): Promise<Episode> {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `run -> ${res.status}`);
  return data as Episode;
}

export interface StreamParams {
  seed?: number;
  horizon?: number;
  max_steps?: number;
  tag?: string;
  mediator?: string;
  pace?: number;
}

export function streamUrl(p: StreamParams): string {
  const q = new URLSearchParams();
  if (p.seed != null) q.set("seed", String(p.seed));
  if (p.horizon != null) q.set("horizon", String(p.horizon));
  if (p.max_steps != null) q.set("max_steps", String(p.max_steps));
  if (p.tag) q.set("tag", p.tag);
  if (p.mediator) q.set("mediator", p.mediator);
  if (p.pace != null) q.set("pace", String(p.pace));
  return `${API_BASE}/api/stream?${q.toString()}`;
}

// A record off the wire is either the meta header or a tick.
export type StreamRecord = Meta | Tick;

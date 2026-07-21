import type { Meta } from "./types";
import { KIND_ORDER } from "./types";

export interface Pos {
  x: number;
  y: number;
}

/** Column-per-kind layout (farm -> hub -> dc -> retail), rows spread vertically. */
export function nodePositions(meta: Meta): Record<string, Pos> {
  const byKind: Record<string, string[]> = {};
  for (const n of meta.nodes) (byKind[n.kind] ??= []).push(n.name);

  const pos: Record<string, Pos> = {};
  const cols = KIND_ORDER.length - 1;
  KIND_ORDER.forEach((kind, col) => {
    const names = (byKind[kind] ?? []).slice().sort();
    const n = names.length;
    names.forEach((name, i) => {
      pos[name] = {
        x: cols === 0 ? 0.5 : col / cols,
        y: n <= 1 ? 0.5 : 0.1 + (0.8 * i) / (n - 1),
      };
    });
  });
  return pos;
}

/** Fan several markers sitting on the same node so they don't overlap. */
export function fanned(pos: Pos, index: number, total: number): Pos {
  if (total <= 1) return pos;
  const spread = 0.05;
  const offset = (index - (total - 1) / 2) * spread;
  return { x: pos.x + offset, y: pos.y + offset };
}

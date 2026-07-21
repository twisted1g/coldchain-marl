import { useCallback, useEffect, useState } from "react";
import { listEpisodes, loadEpisode } from "./api";
import { useLiveStream } from "./hooks/useLiveStream";
import { GraphView } from "./components/GraphView";
import type { HoverTarget } from "./components/GraphView";
import { HoverCard } from "./components/HoverCard";
import { InventoryPanel } from "./components/InventoryPanel";
import { NegotiationPanel } from "./components/NegotiationPanel";
import { SystemBar } from "./components/SystemBar";
import { StepBar } from "./components/StepBar";
import type { Episode, Tick } from "./types";

/**
 * Dashboard shell — components cleared for a rewrite.
 *
 * The data plumbing stays: episode list + loader, live SSE stream, and a
 * current-tick pointer. Drop new components under <main> and feed them `meta`
 * and `tick` (and `visibleTicks` for time-series). See src/api.ts + src/types.ts
 * for the data contracts and src/Plot.tsx for the Plotly wrapper.
 */
export function App() {
  const [episodes, setEpisodes] = useState<string[]>([]);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [mode, setMode] = useState<"episode" | "live">("episode");
  const [index, setIndex] = useState(0);
  const [status, setStatus] = useState("loading…");
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<{
    target: HoverTarget;
    x: number;
    y: number;
  } | null>(null);

  const live = useLiveStream();
  const isLive = mode === "live";

  const meta = isLive ? live.meta : episode?.meta ?? null;
  const ticks = isLive ? live.ticks : episode?.ticks ?? [];
  const clampedIndex = Math.min(index, Math.max(0, ticks.length - 1));
  const tick: Tick | null = ticks[clampedIndex] ?? null;

  const selectEpisode = useCallback(
    async (name: string) => {
      live.stop();
      setMode("episode");
      try {
        const ep = await loadEpisode(name);
        setEpisode(ep);
        setIndex(0);
        setPlaying(false);
        setStatus(`${name} · ${ep.ticks.length} ticks`);
      } catch (e) {
        setStatus(`load failed · ${(e as Error).message}`);
      }
    },
    [live],
  );

  useEffect(() => {
    listEpisodes()
      .then(({ episodes }) => {
        setEpisodes(episodes);
        if (episodes.length) selectEpisode(episodes[episodes.length - 1]);
        else setStatus("no episodes recorded");
      })
      .catch((e) => setStatus(`API offline · ${e.message}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isLive) setIndex(Math.max(0, live.ticks.length - 1));
  }, [isLive, live.ticks.length]);

  // Auto-advance the tick pointer while playing (episode mode only). Stops at
  // the last tick. Every tick moves all agents together — the frame already
  // holds each vehicle's state, so stepping renders them in parallel.
  useEffect(() => {
    if (!playing || isLive || ticks.length === 0) return;
    const id = setInterval(() => {
      setIndex((i) => {
        if (i >= ticks.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 700);
    return () => clearInterval(id);
  }, [playing, isLive, ticks.length]);

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo">❄</span>
          <div>
            <h1>Cold-Chain MARL</h1>
            <span className="sub">cold-chain supply network</span>
          </div>
        </div>
        <div className="controls">
          <label className="field">
            episode
            <select
              value={isLive ? "" : episode?.name ?? ""}
              onChange={(e) => selectEpisode(e.target.value)}
            >
              <option value="" disabled>
                {episodes.length ? "select…" : "none"}
              </option>
              {episodes.map((e) => (
                <option key={e} value={e}>{e}</option>
              ))}
            </select>
          </label>
          <span className="status">{status}</span>
        </div>
      </header>

      {meta ? (
        <>
        <SystemBar tick={tick} />
        <StepBar
          index={clampedIndex}
          count={ticks.length}
          playing={playing}
          onIndex={(i) => {
            setPlaying(false);
            setIndex(i);
          }}
          onTogglePlay={() => setPlaying((p) => !p)}
          tickLabel={
            meta
              ? `tick ${tick?.tick ?? 0} / ${meta.max_steps}`
              : undefined
          }
        />
        <main className="layout">
          <GraphView
            meta={meta}
            tick={tick}
            onHover={(target, x, y) => setHover({ target, x, y })}
            onUnhover={() => setHover(null)}
          />
          <div className="area-side">
            <InventoryPanel tick={tick} />
            <NegotiationPanel tick={tick} />
          </div>
          <HoverCard
            target={hover?.target ?? null}
            pos={hover ? { x: hover.x, y: hover.y } : null}
            meta={meta}
            tick={tick}
          />
        </main>
        </>
      ) : (
        <main className="empty">
          <p className="hint">{status}</p>
        </main>
      )}
    </>
  );
}

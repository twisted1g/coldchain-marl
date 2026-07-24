import { useCallback, useEffect, useRef, useState } from "react";
import { streamUrl, type StreamParams } from "../api";
import type { Meta, Tick } from "../types";

export type LiveStatus = "idle" | "connecting" | "streaming" | "ended" | "error";

interface LiveState {
  meta: Meta | null;
  ticks: Tick[];
  status: LiveStatus;
  error: string | null;
}

const INITIAL: LiveState = { meta: null, ticks: [], status: "idle", error: null };

/** Drive a rolling live inference over SSE, accumulating meta + ticks. */
export function useLiveStream() {
  const [state, setState] = useState<LiveState>(INITIAL);
  const esRef = useRef<EventSource | null>(null);

  const stop = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setState((s) =>
      s.status === "streaming" || s.status === "connecting"
        ? { ...s, status: "ended" }
        : s,
    );
  }, []);

  const start = useCallback((params: StreamParams) => {
    esRef.current?.close();
    setState({ ...INITIAL, status: "connecting" });

    const es = new EventSource(streamUrl(params));
    esRef.current = es;

    es.onmessage = (ev) => {
      const rec = JSON.parse(ev.data) as Meta | Tick;
      setState((s) => {
        if (rec.type === "meta") {
          return { meta: rec, ticks: [], status: "streaming", error: null };
        }
        return { ...s, status: "streaming", ticks: [...s.ticks, rec] };
      });
    };
    es.addEventListener("end", () => {
      es.close();
      esRef.current = null;
      setState((s) => ({ ...s, status: "ended" }));
    });
    es.addEventListener("error", (ev: MessageEvent) => {
      let msg = "stream error";
      try {
        if (ev.data) msg = JSON.parse(ev.data).error ?? msg;
      } catch {
        /* browser reconnect error carries no data */
      }
      es.close();
      esRef.current = null;
      setState((s) => ({ ...s, status: "error", error: msg }));
    });
  }, []);

  useEffect(() => () => esRef.current?.close(), []);

  return { ...state, start, stop };
}

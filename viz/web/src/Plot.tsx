import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, Config } from "plotly.js-dist-min";

interface PlotProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  /** Fires with the clicked point's `customdata` (undefined if none set). */
  onPointClick?: (customdata: unknown) => void;
  /** Fires while hovering a point that carries `customdata`, with viewport px. */
  onPointHover?: (customdata: unknown, x: number, y: number) => void;
  onPointUnhover?: () => void;
}

const BASE_LAYOUT: Partial<Layout> = {
  margin: { l: 40, r: 12, t: 8, b: 28 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, system-ui, sans-serif", size: 11, color: "#334155" },
  showlegend: false,
};

const BASE_CONFIG: Partial<Config> = { displayModeBar: false, responsive: true };

/** Thin declarative wrapper over Plotly.react on a resizing div. */
export function Plot({
  data,
  layout,
  config,
  className,
  onPointClick,
  onPointHover,
  onPointUnhover,
}: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Plotly.react(
      el,
      data,
      { ...BASE_LAYOUT, ...layout },
      { ...BASE_CONFIG, ...config },
    );
    // Plotly divs are EventEmitters; clear stale handlers before re-binding.
    type Ev = { points?: { customdata?: unknown }[]; event?: MouseEvent };
    const gd = el as unknown as {
      on: (e: string, cb: (ev: Ev) => void) => void;
      removeAllListeners?: (e: string) => void;
    };
    if (onPointClick) {
      gd.removeAllListeners?.("plotly_click");
      gd.on("plotly_click", (ev) => {
        const cd = ev.points?.[0]?.customdata;
        if (cd !== undefined) onPointClick(cd);
      });
    }
    if (onPointHover) {
      gd.removeAllListeners?.("plotly_hover");
      gd.on("plotly_hover", (ev) => {
        const cd = ev.points?.[0]?.customdata;
        if (cd !== undefined && ev.event)
          onPointHover(cd, ev.event.clientX, ev.event.clientY);
      });
    }
    if (onPointUnhover) {
      gd.removeAllListeners?.("plotly_unhover");
      gd.on("plotly_unhover", () => onPointUnhover());
    }
  }, [data, layout, config, onPointClick, onPointHover, onPointUnhover]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} className={className} />;
}

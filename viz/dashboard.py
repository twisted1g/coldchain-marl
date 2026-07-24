"""Render a recorded episode (viz.record JSONL) as a world dashboard.

One frame per tick: the supply-chain graph with the shipment, disruptions and
delivery vehicles on the left; inventory, temperature, spoilage and delivery-slot
panels on the right; a header line with tick, weather, calendar, disruptions and
rewards. Outputs per-tick PNG frames plus an animated GIF (and an MP4 if ffmpeg
is available).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

KIND_ORDER = ["farm", "hub", "dc", "retail"]
KIND_COLOR = {
    "farm": "#4c9f70",
    "hub": "#6689c9",
    "dc": "#c9a14a",
    "retail": "#c96666",
}
WEATHER_COLOR = {
    "sunny": "#f5c542",
    "cloudy": "#b0b0b0",
    "rainy": "#5a8fc9",
    "stormy": "#4a4a6a",
}


def load_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    ticks: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "meta":
                meta = rec
            else:
                ticks.append(rec)
    if meta is None:
        raise ValueError(f"no meta record in {path}")
    return meta, ticks


def node_layout(meta: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Tiered layout: kind sets the column, name spreads down the column."""
    by_kind: dict[str, list[str]] = {k: [] for k in KIND_ORDER}
    for node in meta["nodes"]:
        by_kind[node["kind"]].append(node["name"])
    pos: dict[str, tuple[float, float]] = {}
    for col, kind in enumerate(KIND_ORDER):
        names = sorted(by_kind[kind])
        n = len(names)
        for row, name in enumerate(names):
            y = (n - 1) / 2 - row
            pos[name] = (float(col), float(y))
    return pos


class DashboardRenderer:
    def __init__(self, meta: dict[str, Any], ticks: list[dict[str, Any]]) -> None:
        self.meta = meta
        self.ticks = ticks
        self.pos = node_layout(meta)
        self.kind_of = {n["name"]: n["kind"] for n in meta["nodes"]}
        self.retailers = sorted(
            n["name"] for n in meta["nodes"] if n["kind"] == "retail"
        )

        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0])
        self.ax_graph = self.fig.add_subplot(gs[:, 0])
        self.ax_inv = self.fig.add_subplot(gs[0, 1])
        self.ax_temp = self.fig.add_subplot(gs[1, 1])
        self.ax_spoil = self.fig.add_subplot(gs[2, 1])
        self.ax_slots = self.fig.add_subplot(gs[3, 1])
        self.fig.subplots_adjust(
            left=0.02, right=0.97, top=0.90, bottom=0.06, hspace=0.6, wspace=0.12
        )

    # -- header ---------------------------------------------------------------
    def _header(self, t: dict[str, Any]) -> None:
        a = t["ambient"]
        c = t["calendar"]
        disr = ", ".join(f"{d['type']}@{d['target']}" for d in t["disruptions"])
        disr = disr or "none"
        reward = t["rewards"]
        rtxt = "  ".join(f"{k.split('_')[0][:4]}:{v:+.2f}" for k, v in reward.items())
        title = (
            f"{self.meta['fruit'].split('.')[-1]}  "
            f"{self.meta['source']} → {self.meta['target']}   |   "
            f"tick {t['tick']}/{self.meta['max_steps']}   |   "
            f"weather {a['weather']} ({a['temp']:.1f}°C)   |   "
            f"day {c['day_of_year']} wd{c['weekday']} x{c['event_multiplier']:.2f}\n"
            f"disruptions: {disr}"
        )
        self.fig.suptitle(title, fontsize=10, ha="center")
        if rtxt:
            self.fig.text(0.5, 0.925, rtxt, fontsize=8, ha="center", color="#333")

    # -- graph panel ----------------------------------------------------------
    def _draw_graph(self, t: dict[str, Any]) -> None:
        ax = self.ax_graph
        ax.set_title("supply-chain world", fontsize=10)
        disrupted = {d["target"] for d in t["disruptions"]}
        for u, v in self.meta["edges"]:
            (x0, y0), (x1, y1) = self.pos[u], self.pos[v]
            ax.plot([x0, x1], [y0, y1], color="#dcdcdc", lw=0.8, zorder=1)

        ship = t["shipment"]
        for name, (x, y) in self.pos.items():
            kind = self.kind_of[name]
            ax.scatter(x, y, s=520, color=KIND_COLOR[kind], zorder=2,
                       edgecolors="white", linewidths=1.5)
            ax.text(x, y - 0.32, name, fontsize=6.5, ha="center", va="top",
                    color="#333")
            if name in disrupted:
                ax.add_patch(Circle((x, y), 0.22, fill=False, ec="red",
                                    lw=2.0, zorder=4))

        tx, ty = self.pos[ship["target_node"]]
        ax.add_patch(Circle((tx, ty), 0.28, fill=False, ec="#222", lw=1.6,
                            ls="--", zorder=3))
        cx, cy = self.pos[ship["current_node"]]
        risk = ship["spoilage_risk"]
        ax.scatter(cx, cy, s=180, marker="D",
                   color=plt.cm.RdYlGn_r(min(1.0, risk)), zorder=5,
                   edgecolors="black", linewidths=1.0)
        ax.text(cx, cy + 0.34, f"cargo r{risk:.2f}", fontsize=7, ha="center",
                color="#222", zorder=5)

        for i, veh in enumerate(t["vehicles"]):
            rx, ry = self.pos[veh["assigned_node"]]
            offy = 0.42 + 0.18 * i
            flag = "!" if veh["sla_violated"] else ("x" if veh["conflict"] else "")
            col = "red" if veh["sla_violated"] else (
                "orange" if veh["conflict"] else "#333")
            ax.text(rx + 0.30, ry - offy, f"v{i} s{veh['chosen_slot']}{flag}",
                    fontsize=6.5, ha="left", color=col)

        ax.set_xlim(-0.6, 3.9)
        ymax = max(abs(y) for _, y in self.pos.values()) + 1.2
        ax.set_ylim(-ymax, ymax)
        ax.axis("off")

    # -- inventory panel ------------------------------------------------------
    def _draw_inventory(self, t: dict[str, Any]) -> None:
        ax = self.ax_inv
        ax.set_title("inventory / retailer", fontsize=9)
        inv = t["inventory"]
        n = len(self.retailers)
        idx = range(n)
        w = 0.2
        ax.bar([i - 1.5 * w for i in idx], inv["levels"], w, label="stock",
               color="#6689c9")
        ax.bar([i - 0.5 * w for i in idx], inv["order"], w, label="order",
               color="#4c9f70")
        ax.bar([i + 0.5 * w for i in idx], inv["demand_today"], w, label="demand",
               color="#c9a14a")
        ax.bar([i + 1.5 * w for i in idx], inv["unmet"], w, label="unmet",
               color="#c96666")
        ax.set_xticks(list(idx))
        ax.set_xticklabels([r.replace("retail_", "R") for r in self.retailers],
                           fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=4, loc="upper center", framealpha=0.6)

    # -- temperature panel ----------------------------------------------------
    def _draw_temp(self, upto: int) -> None:
        ax = self.ax_temp
        ax.set_title("temperature (°C)", fontsize=9)
        th = self.meta["thresholds"]
        xs = list(range(upto + 1))
        sensor = [self.ticks[i]["shipment"]["sensor_temp"] for i in xs]
        desired = [self.ticks[i]["shipment"]["desired_temp"] for i in xs]
        ambient = [self.ticks[i]["ambient"]["temp"] for i in xs]
        ax.axhspan(th["optimal_temp_low"], th["optimal_temp_high"],
                   color="#9fdf9f", alpha=0.35, label="optimal")
        if th["chill_injury"] is not None:
            ax.axhline(th["chill_injury"], color="#5a8fc9", ls=":", lw=1.0,
                       label="chill")
        ax.plot(xs, sensor, "-o", ms=3, color="#c0392b", label="sensor")
        ax.plot(xs, desired, "--", color="#8e44ad", label="setpoint")
        ax.plot(xs, ambient, ":", color="#888", label="ambient")
        ax.set_xlim(0, self.meta["max_steps"])
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.6)

    # -- spoilage panel -------------------------------------------------------
    def _draw_spoilage(self, upto: int) -> None:
        ax = self.ax_spoil
        ax.set_title("spoilage risk / freshness", fontsize=9)
        xs = list(range(upto + 1))
        risk = [self.ticks[i]["shipment"]["spoilage_risk"] for i in xs]
        fresh = [self.ticks[i]["shipment"]["freshness_score"] for i in xs]
        pred = [self.ticks[i]["spoilage_prediction"] for i in xs]
        ax.plot(xs, risk, "-o", ms=3, color="#c0392b", label="risk")
        ax.plot(xs, fresh, "-", color="#4c9f70", label="freshness")
        ax.plot(xs, pred, ":", color="#8e44ad", label="pred")
        ax.axhline(0.5, color="#888", ls="--", lw=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, self.meta["max_steps"])
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.6)

    # -- delivery slots panel -------------------------------------------------
    def _draw_slots(self, t: dict[str, Any]) -> None:
        ax = self.ax_slots
        ax.set_title("delivery windows", fontsize=9)
        n_win = self.meta["n_windows"]
        max_steps = self.meta["max_steps"]
        for w in range(n_win):
            x0 = w / n_win * max_steps
            x1 = (w + 1) / n_win * max_steps
            ax.axvspan(x0, x1, color="#eee" if w % 2 else "#f7f7f7")
            ax.axvline(x1, color="#ccc", lw=0.8)
        ax.axvline(t["tick"], color="#222", lw=1.4, label="now")
        for i, veh in enumerate(t["vehicles"]):
            slot = veh["chosen_slot"]
            deadline = (slot + 1) / n_win * max_steps
            col = "red" if veh["sla_violated"] else (
                "orange" if veh["conflict"] else "#4c9f70")
            ax.scatter(deadline, i, s=60, color=col, zorder=3)
            ax.text(deadline, i + 0.2, f"v{i}", fontsize=7, ha="center")
        ax.set_ylim(-0.6, len(t["vehicles"]) - 0.4)
        ax.set_xlim(0, max_steps)
        ax.set_yticks([])
        ax.set_xlabel("tick", fontsize=7)
        ax.tick_params(labelsize=7)

    # -- frame ----------------------------------------------------------------
    def draw(self, i: int) -> None:
        for ax in (self.ax_graph, self.ax_inv, self.ax_temp, self.ax_spoil,
                   self.ax_slots):
            ax.clear()
        for txt in list(self.fig.texts):
            txt.remove()
        t = self.ticks[i]
        self._header(t)
        self._draw_graph(t)
        self._draw_inventory(t)
        self._draw_temp(i)
        self._draw_spoilage(i)
        self._draw_slots(t)


def render(episode: Path, out_dir: Path, fps: int, mp4: bool) -> None:
    meta, ticks = load_episode(episode)
    r = DashboardRenderer(meta, ticks)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(ticks)):
        r.draw(i)
        r.fig.savefig(out_dir / f"frame_{i:02d}.png", dpi=110)

    anim = FuncAnimation(r.fig, r.draw, frames=len(ticks), interval=1000 / fps)
    gif_path = out_dir / "episode.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    print(f"wrote {len(ticks)} frames + {gif_path}")

    if mp4:
        try:
            from matplotlib.animation import FFMpegWriter

            anim.save(out_dir / "episode.mp4", writer=FFMpegWriter(fps=fps))
            print(f"wrote {out_dir / 'episode.mp4'}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"mp4 skipped (no ffmpeg?): {exc}")
    plt.close(r.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="JSONL from viz.record")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default alongside the JSONL)")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--mp4", action="store_true", help="also write MP4")
    args = parser.parse_args()

    out_dir = args.out or args.episode.parent / args.episode.stem
    render(args.episode, out_dir, args.fps, args.mp4)


if __name__ == "__main__":
    main()

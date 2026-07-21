"""Web service for the world dashboard: serve the JS frontend and episode data.

Zero extra Python dependencies — stdlib ``http.server`` only. Endpoints:

    GET  /                       -> the single-page frontend
    GET  /static/<file>          -> frontend assets (css, js)
    GET  /api/episodes           -> list recorded episodes
    GET  /api/episode/<name>     -> {meta, ticks} for one episode
    POST /api/run                -> roll out a new episode and return it
                                    body: {seed, tag, scenario, max_steps}

The ``/api/run`` handler imports the rollout lazily (it pulls torch), so listing
and viewing recorded episodes stay light.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
EPISODES_DIR = ARTIFACTS / "episodes"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


def _read_episode(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] | None = None
    ticks: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "meta":
                meta = rec
            else:
                ticks.append(rec)
    return {"name": path.stem, "meta": meta, "ticks": ticks}


def _list_episodes() -> list[str]:
    if not EPISODES_DIR.exists():
        return []
    return sorted(p.stem for p in EPISODES_DIR.glob("*.jsonl"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # quieter console
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._send_file(WEB_DIR / "index.html")
        elif path.startswith("/static/"):
            self._send_file(WEB_DIR / path[len("/static/"):])
        elif path == "/api/stream":
            self._stream(parse_qs(parsed.query))
        elif path == "/api/episodes":
            self._send_json({"episodes": _list_episodes()})
        elif path.startswith("/api/episode/"):
            name = path[len("/api/episode/"):]
            fpath = EPISODES_DIR / f"{name}.jsonl"
            if not fpath.is_file():
                self._send_json({"error": "episode not found"}, 404)
                return
            self._send_json(_read_episode(fpath))
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        try:
            payload = self._run_episode(req)
        except Exception as exc:  # surface rollout errors to the UI
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send_json(payload)

    def _run_episode(self, req: dict[str, Any]) -> dict[str, Any]:
        from viz.record import episode_name, record_episode, write_episode

        seed = int(req.get("seed", 90_000))
        episodes = max(1, int(req.get("episodes", 1) or 1))
        tag = req.get("tag") or None
        scenario = req.get("scenario") or None
        max_steps = req.get("max_steps")
        max_steps = int(max_steps) if max_steps else None
        mediator = req.get("mediator") or "off"

        latest: dict[str, Any] = {}
        for k in range(episodes):
            records = record_episode(seed + k, tag, scenario, max_steps, mediator)
            name = episode_name(seed + k, tag)
            write_episode(records, EPISODES_DIR / f"{name}.jsonl")
            meta = next(r for r in records if r["type"] == "meta")
            ticks = [r for r in records if r["type"] == "tick"]
            latest = {"name": name, "meta": meta, "ticks": ticks}
        return latest

    def _stream(self, q: dict[str, list[str]]) -> None:
        """Server-sent events: run a rolling live inference, one tick per event."""
        from viz.live import DEFAULT_HORIZON, live_stream

        def qint(key: str, default: int) -> int:
            try:
                return int(q.get(key, [""])[0])
            except (ValueError, TypeError):
                return default

        seed = qint("seed", 90_000)
        horizon = qint("horizon", DEFAULT_HORIZON)
        max_steps = qint("max_steps", 0) or None
        tag = (q.get("tag", [""])[0] or None)
        mediator = q.get("mediator", ["llm"])[0] or "llm"
        pace = max(0.0, float(q.get("pace", ["0.9"])[0] or 0.9))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for rec in live_stream(seed, tag, horizon, max_steps, mediator):
                self.wfile.write(f"data: {json.dumps(rec)}\n\n".encode())
                self.wfile.flush()
                time.sleep(pace)
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self.wfile.write(
                    f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode()
                )
                self.wfile.flush()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"cold-chain dashboard on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

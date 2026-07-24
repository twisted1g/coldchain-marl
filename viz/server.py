"""Static frontend server for the dashboard.

Serves the built Vite bundle (``web/dist``) — the inference API is a separate
service (``viz.api``). The frontend reaches the API via ``window.API_BASE``,
injected into ``index.html`` at serve time (``--api-base``, default the same host
on the API port). Run both together with ``python -m viz.serve``.

Build the bundle first: ``npm --prefix viz/web install && npm --prefix viz/web run build``.

    GET  /                -> dist/index.html (with API_BASE injected)
    GET  /<asset>         -> dist/<asset> (hashed js/css, etc.)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
DIST = WEB_DIR / "dist"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


class StaticHandler(BaseHTTPRequestHandler):
    api_base: str = ""

    def log_message(self, *args: Any) -> None:  # quieter console
        pass

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_bytes(b"not found", "text/plain; charset=utf-8", 404)
            return
        ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), ctype)

    def _send_index(self) -> None:
        """Serve index.html with the API base URL injected as ``window.API_BASE``."""
        index = DIST / "index.html"
        if not index.is_file():
            self._send_bytes(
                b"frontend not built. run: npm --prefix viz/web install && "
                b"npm --prefix viz/web run build",
                "text/plain; charset=utf-8",
                503,
            )
            return
        html = index.read_text()
        inject = f'<script>window.API_BASE = "{self.api_base}";</script>'
        if "</head>" in html:
            html = html.replace("</head>", f"  {inject}\n</head>", 1)
        else:
            html = inject + html
        self._send_bytes(html.encode(), _CONTENT_TYPES[".html"])

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_index()
            return
        # Serve any built asset by path, guarding against traversal outside dist.
        target = (DIST / path.lstrip("/")).resolve()
        if DIST.resolve() in target.parents:
            self._send_file(target)
        else:
            self._send_bytes(b"not found", "text/plain; charset=utf-8", 404)


def build_server(host: str, port: int, api_base: str) -> ThreadingHTTPServer:
    handler = type("BoundStaticHandler", (StaticHandler,), {"api_base": api_base})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-base",
        default=None,
        help="inference API base URL (default http://<host>:8001)",
    )
    args = parser.parse_args()

    api_base = args.api_base or f"http://{args.host}:8001"
    server = build_server(args.host, args.port, api_base)
    print(f"cold-chain dashboard on http://{args.host}:{args.port}  (API {api_base})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

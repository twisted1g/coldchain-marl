"""Launch both dashboard services together: the inference API and the static
frontend. Convenience for local dev — in production the two run independently
(``python -m viz.api`` and ``python -m viz.server``, possibly on different hosts).
"""

from __future__ import annotations

import argparse
import threading

from viz import api, server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--api-port", type=int, default=8001)
    args = parser.parse_args()

    api_base = f"http://{args.host}:{args.api_port}"
    api_srv = api.build_server(args.host, args.api_port)
    web_srv = server.build_server(args.host, args.web_port, api_base)

    threading.Thread(target=api_srv.serve_forever, daemon=True).start()
    print(f"cold-chain inference API on {api_base}")
    print(f"cold-chain dashboard on http://{args.host}:{args.web_port}")
    try:
        web_srv.serve_forever()
    except KeyboardInterrupt:
        web_srv.shutdown()
        api_srv.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Serve a local browser UI for the pi05_droid remote demo stream."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import urllib.parse
import webbrowser


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pi05_droid Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --line: #d9dee5;
      --text: #17202a;
      --muted: #677383;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #15803d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.8fr);
      gap: 16px;
      padding: 16px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input {
      flex: 1 1 360px;
      min-width: 240px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      min-width: 88px;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: white;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.live { background: var(--good); }
    .dot.error { background: var(--bad); }
    .image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #eef1f4;
      overflow: hidden;
      min-height: 280px;
    }
    figure img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #111827;
    }
    figcaption {
      padding: 8px 10px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      background: #fff;
    }
    .kv {
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 0;
      border-top: 1px solid var(--line);
    }
    .kv div {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      min-height: 36px;
      overflow-wrap: anywhere;
    }
    .kv div:nth-child(odd) {
      color: var(--muted);
      background: #fafbfc;
    }
    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      max-height: 360px;
      border-top: 1px solid var(--line);
      background: #111827;
      color: #e5e7eb;
      font-size: 12px;
    }
    .side-title {
      padding: 12px;
      font-weight: 650;
      border-bottom: 1px solid var(--line);
    }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
      .image-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>pi05_droid Demo</h1>
    <span class="status"><span id="status-dot" class="dot"></span><span id="status-text">idle</span></span>
  </header>
  <main>
    <section>
      <div class="toolbar">
        <input id="ws-url" aria-label="WebSocket URL">
        <button id="connect" class="primary">Connect</button>
        <button id="disconnect">Stop</button>
      </div>
      <div class="image-grid">
        <figure>
          <img id="exterior" alt="Exterior model input">
          <figcaption>Exterior input</figcaption>
        </figure>
        <figure>
          <img id="wrist" alt="Wrist model input">
          <figcaption>Wrist input</figcaption>
        </figure>
      </div>
    </section>
    <aside>
      <div class="side-title">Run State</div>
      <div class="kv">
        <div>Run</div><div id="run-id">-</div>
        <div>Prompt</div><div id="prompt">-</div>
        <div>Frame</div><div id="frame">-</div>
        <div>Queries</div><div id="queries">-</div>
        <div>Roundtrip</div><div id="roundtrip">-</div>
        <div>Server infer</div><div id="server-infer">-</div>
        <div>Action</div><div id="action">-</div>
      </div>
      <div class="side-title">Final Metrics</div>
      <pre id="metrics">{}</pre>
    </aside>
  </main>
  <script>
    const defaultWs = __REMOTE_WS__;
    const params = new URLSearchParams(window.location.search);
    const wsInput = document.getElementById("ws-url");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const metricsEl = document.getElementById("metrics");
    let socket = null;
    let finished = false;

    wsInput.value = params.get("ws") || defaultWs;

    function setStatus(text, kind) {
      statusText.textContent = text;
      statusDot.className = "dot" + (kind ? " " + kind : "");
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value ?? "-";
    }

    function formatMs(stats) {
      if (!stats || stats.mean === undefined) return "-";
      return `${stats.mean.toFixed(1)} ms mean / ${stats.p90.toFixed(1)} ms p90`;
    }

    function setMetrics(metrics) {
      metricsEl.textContent = JSON.stringify(metrics || {}, null, 2);
      setText("queries", metrics?.query_count ?? "-");
      setText("roundtrip", formatMs(metrics?.client_roundtrip_ms));
      setText("server-infer", formatMs(metrics?.server_infer_ms));
    }

    function connect() {
      if (socket) socket.close();
      finished = false;
      setStatus("connecting", "");
      const url = wsInput.value.trim();
      socket = new WebSocket(url);
      socket.onopen = () => setStatus("live", "live");
      socket.onclose = () => {
        if (!finished) setStatus("closed", "");
      };
      socket.onerror = () => setStatus("error", "error");
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data);
        if (event.type === "run_started") {
          setText("run-id", event.run_id);
          setText("prompt", event.episode?.prompt);
          setMetrics({});
        } else if (event.type === "action_chunk") {
          setText("queries", event.query_index);
          setText("roundtrip", `${event.client_roundtrip_ms.toFixed(1)} ms`);
        } else if (event.type === "frame") {
          document.getElementById("exterior").src = `data:image/jpeg;base64,${event.exterior_jpeg}`;
          document.getElementById("wrist").src = `data:image/jpeg;base64,${event.wrist_jpeg}`;
          setText("frame", `${event.frame_index} (${event.frame_count})`);
          setText("prompt", event.prompt);
          setText("action", JSON.stringify((event.action || []).slice(0, 8)));
          setMetrics(event.metrics);
        } else if (event.type === "run_finished") {
          finished = true;
          setStatus("finished", "live");
          setMetrics(event.metrics);
        } else if (event.type === "error") {
          setStatus("error", "error");
          setMetrics(event);
        }
      };
    }

    document.getElementById("connect").addEventListener("click", connect);
    document.getElementById("disconnect").addEventListener("click", () => {
      if (socket) socket.close();
    });
    connect();
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    remote_ws = "ws://127.0.0.1:8765/ws"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_text("OK\n", content_type="text/plain")
            return
        if parsed.path not in ("", "/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        html = HTML_TEMPLATE.replace("__REMOTE_WS__", json.dumps(self.remote_ws))
        self._send_text(html, content_type="text/html; charset=utf-8")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, body: str, *, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ws", default="ws://127.0.0.1:8765/ws", help="Remote demo WebSocket URL")
    parser.add_argument("--host", default="127.0.0.1", help="Local HTTP host")
    parser.add_argument("--port", type=int, default=7860, help="Local HTTP port")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ViewerHandler.remote_ws = args.remote_ws
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Viewer: {url}")
    print(f"Remote WebSocket: {args.remote_ws}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

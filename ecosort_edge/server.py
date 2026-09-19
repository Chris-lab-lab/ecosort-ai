"""Dependency-free JSON API used by the EcoSort mobile application."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from .state import BIN_CATEGORIES, EdgeStateStore


class EdgeApiServer:
    def __init__(
        self,
        state: EdgeStateStore,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self._state = state
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self._server.daemon_threads = True
        self._thread: Thread | None = None

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "EdgeApiServer":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = Thread(
            target=self._server.serve_forever,
            name="ecosort-api",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=3.0)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "EdgeApiServer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        state = self._state

        class Handler(BaseHTTPRequestHandler):
            server_version = "EcoSortEdge/1.0"

            def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self.send_response(HTTPStatus.NO_CONTENT)
                self._cors_headers()
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                parsed = urlparse(self.path)
                if parsed.path == "/api/status":
                    self._json(HTTPStatus.OK, state.status_payload())
                    return
                if parsed.path == "/api/bins":
                    self._json(HTTPStatus.OK, state.bins_payload())
                    return
                if parsed.path == "/api/events":
                    query = parse_qs(parsed.query)
                    try:
                        limit = int(query.get("limit", ["50"])[0])
                    except ValueError:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
                        return
                    since = query.get("since", [None])[0]
                    self._json(
                        HTTPStatus.OK,
                        state.events_payload(limit=limit, since=since),
                    )
                    return
                if parsed.path == "/":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "service": "EcoSort Edge API",
                            "endpoints": ["/api/status", "/api/bins", "/api/events"],
                        },
                    )
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                parsed = urlparse(self.path)
                parts = parsed.path.strip("/").split("/")
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "bins"]
                    and parts[3] == "emptied"
                ):
                    category = parts[2]
                    if category not in BIN_CATEGORIES:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown bin"})
                        return
                    bin_state, event = state.mark_emptied(category)
                    self._json(HTTPStatus.OK, {"bin": bin_state, "event": event})
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(encoded)

            def _cors_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")

        return Handler

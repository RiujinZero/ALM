"""Live browser replay for the ALM-64MB-1C echolocation smoke."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import webbrowser
from collections import OrderedDict
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from manifoldflow.experiments.dendritic_basin_echolocation_smoke import _cupy
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
    run_seed,
)

STATIC_FILE = (
    Path(__file__).resolve().parents[1]
    / "static"
    / "alm_echolocation_visualizer.html"
)


class VisualizerServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(
            error, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)
        ):
            return
        super().handle_error(request, client_address)


def training_args() -> argparse.Namespace:
    """Return the production smoke configuration represented by the visualizer."""
    return argparse.Namespace(
        train_episodes=600,
        candidate_episodes=120,
        test_episodes=240,
        steps=40,
        hidden_steps=24,
        velocity_wander=0.03,
        visual_capacity=1024,
        auditory_capacity=256,
        integration_capacity=1024,
        growth_rounds=8,
        nursery_size=256,
        hill_steps=64,
        fit_samples=6000,
        max_body_bytes=DEFAULT_BODY_BYTES,
        columns=1,
        report=None,
    )


class VisualizerState:
    """Thread-safe live event fan-out with enough retained state for replay."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._latest: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._running = False
        self._predictor: (
            Callable[[float, str, bool, float], dict[str, Any]] | None
        ) = None
        self._inference_lock = threading.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "event"))
        if event_type == "growth":
            key = f"growth:{event.get('organ', 'unknown')}"
        elif event_type == "prediction":
            key = f"prediction:{event.get('stage', 'unknown')}"
        else:
            key = event_type
        with self._lock:
            if event_type == "reset":
                self._latest.clear()
            else:
                self._latest[key] = event
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
            retained = list(self._latest.values())
        subscriber.put({"type": "reset"})
        for event in retained:
            subscriber.put(event)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def start(self, seed: int) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._predictor = None
            self._latest.clear()
        self.publish({"type": "status", "state": "starting", "seed": seed})
        worker = threading.Thread(
            target=self._train,
            args=(seed,),
            name="alm-echolocation-training",
            daemon=True,
        )
        worker.start()
        return True

    def reset(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._latest.clear()
            self._predictor = None
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put({"type": "reset"})
        return True

    def set_predictor(
        self, predictor: Callable[[float, str, bool, float], dict[str, Any]]
    ) -> None:
        with self._lock:
            self._predictor = predictor

    def retarget(
        self,
        target_position: float,
        episode_name: str,
        learning_enabled: bool,
        velocity_scale: float = 1.0,
    ) -> bool:
        with self._lock:
            predictor = self._predictor
            running = self._running
        if predictor is None or running:
            return False
        with self._inference_lock:
            event = predictor(
                target_position, episode_name, learning_enabled, velocity_scale
            )
        self.publish(event)
        return True

    def _train(self, seed: int) -> None:
        try:
            cp = _cupy()
            self.publish({"type": "status", "state": "training", "seed": seed})
            report = run_seed(
                training_args(),
                seed,
                cp,
                event_callback=self.publish,
                model_callback=self.set_predictor,
            )
            self.publish({"type": "complete", "seed": seed, "report": report})
        except Exception as exc:  # surfaced to the local browser instead of lost
            self.publish(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            with self._lock:
                self._running = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"running": self._running, "events": list(self._latest.values())}


def handler_factory(state: VisualizerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ALMEcholocationVisualizer/1.0"

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body = STATIC_FILE.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/status":
                self._json(state.snapshot())
                return
            if path == "/events":
                self._events()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _events(self) -> None:
            subscriber = state.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                        payload = json.dumps(event, separators=(",", ":"))
                        self.wfile.write(f"data: {payload}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            finally:
                state.unsubscribe(subscriber)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/target":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    target_position = float(payload["target_position"])
                    episode_name = str(payload.get("episode_name", "dark_validation"))
                    learning_enabled = bool(payload.get("learning_enabled", False))
                    velocity_scale = max(
                        0.25, min(4.0, float(payload.get("velocity_scale", 1.0)))
                    )
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    self._json(
                        {"ok": False, "message": "Target must be a number."},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                if state.retarget(
                    target_position,
                    episode_name,
                    learning_enabled,
                    velocity_scale,
                ):
                    self._json({"ok": True, "target_position": target_position})
                else:
                    self._json(
                        {
                            "ok": False,
                            "message": "Train an organism before moving the target.",
                        },
                        HTTPStatus.CONFLICT,
                    )
                return
            if path == "/reset":
                if state.reset():
                    self._json({"ok": True})
                else:
                    self._json(
                        {
                            "ok": False,
                            "message": "Wait for the active training run to finish.",
                        },
                        HTTPStatus.CONFLICT,
                    )
                return
            if path != "/start":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                seed = int(payload.get("seed", 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                self._json(
                    {"ok": False, "message": "Seed must be an integer."},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if state.start(seed):
                self._json({"ok": True, "seed": seed}, HTTPStatus.ACCEPTED)
            else:
                self._json(
                    {"ok": False, "message": "A training run is already active."},
                    HTTPStatus.CONFLICT,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = VisualizerState()
    server = VisualizerServer((args.host, args.port), handler_factory(state))
    url = f"http://{args.host}:{server.server_port}"
    print(f"ALM echolocation visualizer: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

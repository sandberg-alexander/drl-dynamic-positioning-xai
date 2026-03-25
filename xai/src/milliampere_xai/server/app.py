"""FastAPI application with WebSocket endpoint and static file serving.

Serves the browser-based XAI dashboard at ``/`` and provides a WebSocket
at ``/ws`` for real-time frame streaming (server -> client) and keyboard
input (client -> server).
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import threading
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from milliampere_xai.server.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# Default location for built web assets (relative to project root)
_DEFAULT_DIST = pathlib.Path(__file__).resolve().parents[4] / "web" / "dist"
# Docker fallback
_DOCKER_DIST = pathlib.Path("/app/web/dist")


def _find_dist_dir(dist_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve the web/dist directory, checking multiple locations."""
    if dist_dir is not None and dist_dir.is_dir():
        return dist_dir
    if _DEFAULT_DIST.is_dir():
        return _DEFAULT_DIST
    if _DOCKER_DIST.is_dir():
        return _DOCKER_DIST
    msg = (
        f"web/dist not found at {_DEFAULT_DIST} or {_DOCKER_DIST}. "
        "Run 'just web-build' first, or pass --dist-dir."
    )
    raise FileNotFoundError(msg)


def create_app(
    on_keyboard: Callable[[int], None] | None = None,
    dist_dir: pathlib.Path | None = None,
) -> tuple[FastAPI, ConnectionManager]:
    """Create FastAPI app with WebSocket endpoint and static file serving.

    Parameters
    ----------
    on_keyboard:
        Callback invoked with mode int (0-5) when the browser sends a
        keyboard event.  In the full XAI pipeline this publishes to
        the ROS ``/drl/mode`` topic.
    dist_dir:
        Override path to the ``web/dist/`` build output.

    Returns
    -------
    tuple of (FastAPI app, ConnectionManager)
    """
    resolved_dist = _find_dist_dir(dist_dir)
    manager = ConnectionManager(on_keyboard=on_keyboard)
    app = FastAPI(title="milliAmpere XAI Dashboard")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                data = await ws.receive_bytes()
                # Client sends msgpack-encoded KeyboardInput
                try:
                    import msgpack

                    msg = msgpack.unpackb(data)
                    key = msg.get("key") if isinstance(msg, dict) else None
                    if key is not None and 0 <= key <= 5:
                        manager.handle_keyboard(int(key))
                    else:
                        logger.warning("Invalid keyboard input: %s", msg)
                except Exception:
                    logger.exception("Failed to decode keyboard input")
        except WebSocketDisconnect:
            manager.disconnect(ws)

    # Serve index.html at root, static assets alongside it
    index_html = resolved_dist / "index.html"

    @app.get("/")
    async def serve_index() -> FileResponse:
        return FileResponse(index_html, media_type="text/html")

    # Mount remaining static assets (JS, CSS, etc.)
    app.mount("/", StaticFiles(directory=str(resolved_dist)), name="static")

    return app, manager


class WebServer:
    """Manages the FastAPI/uvicorn server running in a background thread.

    Provides a synchronous ``broadcast_frame`` method that bridges the
    sync ROS main loop to the async WebSocket broadcast.
    """

    def __init__(
        self,
        on_keyboard: Callable[[int], None] | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        dist_dir: pathlib.Path | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._app, self._manager = create_app(
            on_keyboard=on_keyboard, dist_dir=dist_dir
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def client_count(self) -> int:
        return self._manager.client_count

    def start(self) -> None:
        """Start uvicorn in a daemon thread."""
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(server.serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="xai-web")
        self._thread.start()
        logger.info("Web server started at http://%s:%d", self._host, self._port)

    def broadcast_frame(self, data: bytes) -> None:
        """Send a binary frame to all connected clients (thread-safe).

        Safe to call from the synchronous ROS main loop. If no clients
        are connected or the event loop is not running, this is a no-op.
        """
        if self._loop is None or self._manager.client_count == 0:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._manager.broadcast_bytes(data),
                self._loop,
            )
        except RuntimeError:
            pass  # Event loop closed during shutdown


async def _demo_loop(manager: ConnectionManager, fps: float = 5.0) -> None:
    """Send synthetic RenderFrame data for demo/testing without ROS."""
    import math

    import msgpack
    from milliampere_dp.rendering.frames import (
        ActuatorState,
        RenderFrame,
        ShapFrame,
        VesselState,
    )

    step = 0
    while True:
        t = step * 0.2
        frame = RenderFrame(
            vessel=VesselState(
                x_tilde=2.0 * math.sin(t * 0.3),
                y_tilde=1.5 * math.cos(t * 0.2),
                psi_tilde=10.0 * math.sin(t * 0.1),
                u_hat=0.1 * math.sin(t * 0.5),
                v_hat=0.05 * math.cos(t * 0.4),
                r_hat=0.5 * math.sin(t * 0.15),
                target_pose=(10.0, 20.0, 45.0),
                epsilon_ned=(
                    2.0 * math.sin(t * 0.3),
                    1.5 * math.cos(t * 0.2),
                    10.0 * math.sin(t * 0.1),
                ),
            ),
            actuators=ActuatorState(
                actuator_ref=[
                    (0.5 + 0.3 * math.sin(t * 0.4 + i), 30.0 * math.sin(t * 0.2 + i))
                    for i in range(4)
                ],
                tot_thrust=1.5 + math.sin(t * 0.3),
                tot_angle=20.0 * math.sin(t * 0.15),
                tot_angular_thrust=0.3 * math.cos(t * 0.2),
            ),
            shap=ShapFrame(
                shap_values_action=[
                    [0.1 * math.sin(t * 0.1 + f * 0.5 + o) for f in range(14)]
                    for o in range(8)
                ],
                shap_values_value=[
                    0.05 * math.cos(t * 0.2 + f * 0.3) for f in range(14)
                ],
                base_vectors=[0.0] * 8,
                action_low=[-1.0] * 8,
                action_high=[1.0] * 8,
            ),
            time_step=step,
            time_seconds=t,
        )
        data: bytes = msgpack.packb(frame.model_dump())  # type: ignore[assignment]
        await manager.broadcast_bytes(data)
        step += 1
        await asyncio.sleep(1.0 / fps)


def main() -> None:
    """Standalone entry point: serve the web dashboard without ROS/SHAP."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="XAI web dashboard server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to serve on (default: 8080)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--dist-dir",
        type=pathlib.Path,
        default=None,
        help="Override path to web/dist/ build output",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Send synthetic data for testing (no ROS needed)",
    )
    args = parser.parse_args()

    _demo_task = None

    app, manager = create_app(dist_dir=args.dist_dir)

    if args.demo:

        @app.on_event("startup")  # type: ignore[deprecated]
        async def _start_demo() -> None:
            nonlocal _demo_task
            _demo_task = asyncio.create_task(_demo_loop(manager))

    print(f"Serving XAI dashboard at http://{args.host}:{args.port}")
    if args.demo:
        print("Demo mode: broadcasting synthetic frames at 5 FPS")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""WebSocket connection tracking and frame broadcasting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Track active WebSocket connections and broadcast binary frames.

    Thread-safe for the broadcast path: the shap_explainer main loop
    calls ``broadcast_bytes`` via ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        on_keyboard: Callable[[int], None] | None = None,
    ) -> None:
        self._connections: list[WebSocket] = []
        self._on_keyboard = on_keyboard

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("Client connected (%d active)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("Client disconnected (%d active)", len(self._connections))

    async def broadcast_bytes(self, data: bytes) -> None:
        """Send binary payload to all connected clients.

        Disconnected clients are silently removed.
        """
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_bytes(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

    def handle_keyboard(self, key: int) -> None:
        """Route keyboard input from browser to ROS publisher."""
        if self._on_keyboard is not None:
            self._on_keyboard(key)

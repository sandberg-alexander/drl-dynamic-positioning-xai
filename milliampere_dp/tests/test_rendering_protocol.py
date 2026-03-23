"""Tests for milliampere_dp.rendering.protocol."""

from __future__ import annotations

import pytest

from milliampere_dp.rendering.protocol import Renderer


class TestRendererProtocol:
    def test_runtime_checkable(self):
        """Protocol is decorated with @runtime_checkable."""
        assert hasattr(Renderer, "__protocol_attrs__") or hasattr(
            Renderer, "__abstractmethods__"
        )

    def test_pygame_renderer_satisfies_protocol(self):
        pytest.importorskip("pygame")
        from milliampere_dp.rendering.pygame_renderer import PygameRenderer

        renderer = PygameRenderer()
        assert isinstance(renderer, Renderer)

    def test_non_conformant_fails(self):
        """A plain object does not satisfy the Renderer protocol."""

        class NotARenderer:
            pass

        assert not isinstance(NotARenderer(), Renderer)

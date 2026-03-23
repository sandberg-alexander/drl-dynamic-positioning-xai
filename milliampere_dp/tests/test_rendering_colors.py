"""Tests for milliampere_dp.rendering.colors."""

from __future__ import annotations

from milliampere_dp.rendering.colors import Color


class TestColor:
    def test_all_rgb_or_rgba(self):
        """Every color value must be a 3-tuple or 4-tuple of ints in [0, 255]."""
        for c in Color:
            assert len(c.value) in (3, 4), f"{c.name} has {len(c.value)} channels"
            for channel in c.value:
                assert isinstance(channel, int), f"{c.name} channel {channel} not int"
                assert 0 <= channel <= 255, f"{c.name} channel {channel} out of range"

    def test_known_values(self):
        assert Color.WHITE.value == (255, 255, 255)
        assert Color.BLACK.value == (0, 0, 0)
        assert Color.RED.value == (255, 0, 0)
        assert Color.GREEN.value == (0, 255, 0)

    def test_rgba_colors(self):
        assert len(Color.AGENT_BLUE.value) == 4
        assert len(Color.LEGEND_BOX.value) == 4

    def test_count(self):
        assert len(Color) == 21

    def test_hex_property_rgb(self):
        assert Color.WHITE.hex == "#ffffff"
        assert Color.BLACK.hex == "#000000"
        assert Color.RED.hex == "#ff0000"

    def test_hex_property_rgba(self):
        """Hex uses only the first 3 channels."""
        assert Color.AGENT_BLUE.hex == "#007aff"

    def test_hex_format(self):
        """All hex values should be 7-char strings starting with #."""
        for c in Color:
            h = c.hex
            assert h.startswith("#")
            assert len(h) == 7

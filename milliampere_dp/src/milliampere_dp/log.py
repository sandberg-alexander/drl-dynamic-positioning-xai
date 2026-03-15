"""Structured logging for milliampere_dp.

Provides a common logging configuration that works on the host (pure Python)
and can be bridged to ROS logging when running inside a ROS node by adding
a custom handler that forwards records to rospy.loginfo/logwarn/etc.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured format.

    Safe to call multiple times -- only configures once.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the milliampere_dp namespace.

    Calls setup_logging() automatically if not already configured.
    """
    setup_logging()
    return logging.getLogger(f"milliampere_dp.{name}")

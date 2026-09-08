"""
events.py
Shared event and intent dataclasses used across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class Modality(str, Enum):
    GESTURE = "gesture"
    VOICE   = "voice"
    SCREEN  = "screen"


@dataclass(frozen=True)
class ContextState:
    """Snapshot of the active application context."""
    app_key:      str | None
    window_title: str
    process_name: str
    category:     str
    timestamp:    float = field(default_factory=time.time)


@dataclass(frozen=True)
class GestureEvent:
    """A single classified gesture frame."""
    gesture:    str
    confidence: float
    details:    dict
    modality:   Modality = Modality.GESTURE
    timestamp:  float    = field(default_factory=time.time)


@dataclass(frozen=True)
class Intent:
    """
    A context-resolved decision produced by the intent classifier.

    action is an action_executor-compatible string for fire-once actions,
    or None for stateful intents (scroll, drag) handled by the caller.
    """
    name:    str
    action:  str | None
    label:   str
    context: str
    source:  GestureEvent | None = None

"""
events.py

Shared event/intent vocabulary for the context-aware assistant.

Today only gestures produce events, but every future modality (voice
commands, screen-OCR triggers) should emit into these same shapes so a
single IntentClassifier can reason over all of them. Keeping this as
plain dataclasses (no MediaPipe/pyautogui imports) means it stays cheap
to import from anywhere — voice_pipeline.py, screen_reader.py, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class Modality(str, Enum):
    GESTURE = "gesture"
    VOICE = "voice"      # reserved for weeks 8-9
    SCREEN = "screen"    # reserved for the OCR/screen-understanding module


@dataclass(frozen=True)
class ContextState:
    """Snapshot of 'what app/context is the user in right now'."""
    app_key: str | None          # matched key from config.yaml `apps:` section, or None
    window_title: str
    process_name: str
    category: str                 # coarse bucket: browser / video_player / code_editor / default
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GestureEvent:
    """A single classified gesture frame from gesture_detector.py."""
    gesture: str
    confidence: float
    details: dict
    modality: Modality = Modality.GESTURE
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Intent:
    """
    The output of the IntentClassifier: a context-resolved decision about
    what should happen, independent of *how* it gets executed.

    `action` is an action_executor.py-compatible string (e.g. "right_click",
    "volume_up", "key:alt+left") when the intent maps to a single fire-once
    action. For intents that need continuous/stateful handling in the main
    loop (scrolling, seeking, dragging), `action` is None and the caller
    switches on `name` instead.
    """
    name: str            # e.g. "right_click", "volume_up", "scroll", "video_seek", "browser_nav", "pause_toggle"
    action: str | None   # action_executor-ready string, or None if handled by caller
    label: str            # human-readable, shown on the HUD overlay
    context: str           # which context bucket produced this
    source: GestureEvent | None = None
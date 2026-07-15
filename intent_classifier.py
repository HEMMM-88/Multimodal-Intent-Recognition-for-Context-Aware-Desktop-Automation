"""
intent_classifier.py

This is the thesis-relevant piece: instead of the old approach of scattering
`if app_context == "video_player": ...` checks across the main loop, every
context-dependent decision lives in one rule table here. main.py asks
"given this gesture and this context, what's the intent?" and gets back a
single answer.

This module is intentionally modality-agnostic in *shape* (it consumes a
gesture name + context string), so the same classify() contract can later
accept a normalized voice command or a screen-detected trigger without
changing its interface — only the rule table grows.

Swap-out path for the optional AI-planner stretch goal: replace the
dict lookup in `classify()` with a call into an LLM-backed planner that
returns the same Intent shape, no other module needs to change.
"""

from __future__ import annotations

from events import Intent, GestureEvent

# Fallback context bucket used when no context-specific rule exists.
DEFAULT_CONTEXT = "default"

# ── Rule table ────────────────────────────────────────────────────────────
# context -> gesture -> (intent_name, action_or_None, label)
# Only entries that *differ* from "default" need to be listed per context;
# classify() falls back to "default" automatically.
RULES: dict[str, dict[str, tuple[str, str | None, str]]] = {
    "default": {
        "two_finger_tap": ("right_click", "right_click", "RIGHT_CLICK"),
        "thumbs_up": ("volume_up", "volume_up", "VOLUME_UP"),
        "thumbs_down": ("volume_down", "volume_down", "VOLUME_DOWN"),
        "closed_fist": ("pause_toggle", None, "PAUSE_TOGGLE"),
        "three_fingers_up": ("scroll", None, "SCROLL"),
        "open_palm_horizontal": ("none", None, "-"),
        "primary_click_swipe": ("none", None, "-"),
    },
    "browser": {
        "primary_click_swipe": ("browser_nav", None, "BACK/FORWARD"),
    },
    "video_player": {
        "three_fingers_up": ("volume_step", None, "VOLUME_STEP"),
        "open_palm_horizontal": ("video_seek", None, "SEEK"),
    },
    "code_editor": {
        "two_finger_tap": ("right_click", "right_click", "CONTEXT_MENU"),
    },
}


def classify(gesture: str, context: str, event: GestureEvent | None = None) -> Intent:
    """
    Resolve a (gesture, context) pair into an Intent.

    `gesture` may be a real gesture name ("two_finger_tap", "thumbs_up", ...)
    or one of the synthetic trigger keys used for motion-based decisions
    ("primary_click_swipe", "open_palm_horizontal") — main.py raises those
    when the relevant motion threshold is crossed, since the classifier
    itself is stateless and doesn't track hold-timers or EMA positions.
    """
    context_rules = RULES.get(context, {})
    default_rules = RULES[DEFAULT_CONTEXT]

    name, action, label = context_rules.get(
        gesture, default_rules.get(gesture, ("none", None, "-"))
    )
    return Intent(name=name, action=action, label=label, context=context, source=event)


def register_rule(context: str, gesture: str, intent_name: str, action: str | None, label: str) -> None:
    """
    Runtime hook for config.yaml-driven overrides or a future admin UI —
    lets new context/gesture rules be added without editing this file.
    """
    RULES.setdefault(context, {})[gesture] = (intent_name, action, label)
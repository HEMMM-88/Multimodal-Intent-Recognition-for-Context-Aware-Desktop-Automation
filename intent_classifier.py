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
        "two_hand_swipe": ("window_switch", None, "WINDOW_SWITCH"),
    },
    "browser": {
        "primary_click_swipe": ("browser_nav", None, "BACK/FORWARD"),
        "two_hand_swipe": ("browser_tab_switch", None, "TAB_SWITCH"),
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


def load_rules_from_config(config: dict) -> int:
    """
    Merge config.yaml's `context_rules:` section into RULES, overriding the
    built-in defaults where present. Malformed entries are logged and
    skipped rather than crashing startup — a typo in the YAML shouldn't
    take down the whole assistant.

    Returns the number of rules successfully loaded.
    """
    import logging
    logger = logging.getLogger(__name__)

    context_rules = config.get("context_rules")
    if not context_rules:
        logger.info("No context_rules in config — using built-in defaults only.")
        return 0

    loaded = 0
    for context, gestures in context_rules.items():
        if not isinstance(gestures, dict):
            logger.warning("context_rules.%s is not a mapping — skipped.", context)
            continue
        for gesture, spec in gestures.items():
            if not isinstance(spec, dict) or "intent" not in spec or "label" not in spec:
                logger.warning(
                    "context_rules.%s.%s missing required 'intent'/'label' — skipped.",
                    context, gesture,
                )
                continue
            register_rule(
                context=context,
                gesture=gesture,
                intent_name=spec["intent"],
                action=spec.get("action"),
                label=spec["label"],
            )
            loaded += 1

    logger.info("Loaded %d context rule(s) from config.yaml.", loaded)
    return loaded


# ── Two-hand combo rules ────────────────────────────────────────────────────
# context -> primary_gesture -> (intent_name, action, label)
#
# Deliberately simple: a combo fires whenever a second hand is present at
# all (any pose) while the primary hand does one of the eligible gestures —
# there's no modifier-hand pose to get right. This was originally keyed on
# (modifier_pose, primary_gesture) pairs, but that asked people to remember
# which pose meant what, which is exactly the kind of thing that's easy to
# get right in a spec and hard to actually do in front of a webcam. "Is my
# other hand up or not" is a much lower bar.
COMBO_RULES: dict[str, dict[str, tuple[str, str | None, str]]] = {}


def register_combo_rule(
    context: str, primary_gesture: str, intent_name: str, action: str | None, label: str,
) -> None:
    """Runtime hook mirroring register_rule(), for combo entries."""
    COMBO_RULES.setdefault(context, {})[primary_gesture] = (intent_name, action, label)


def load_combo_rules_from_config(config: dict) -> int:
    """
    Merge config.yaml's `combo_rules:` section into COMBO_RULES. Same
    fail-soft behavior as load_rules_from_config(): bad entries are logged
    and skipped, never fatal.
    """
    import logging
    logger = logging.getLogger(__name__)

    combo_rules = config.get("combo_rules")
    if not combo_rules:
        logger.info("No combo_rules in config — two-hand combos disabled.")
        return 0

    loaded = 0
    for context, gestures in combo_rules.items():
        if not isinstance(gestures, dict):
            logger.warning("combo_rules.%s is not a mapping — skipped.", context)
            continue
        for primary_gesture, spec in gestures.items():
            if not isinstance(spec, dict) or "intent" not in spec or "label" not in spec:
                logger.warning(
                    "combo_rules.%s.%s missing required 'intent'/'label' — skipped.",
                    context, primary_gesture,
                )
                continue
            register_combo_rule(
                context=context,
                primary_gesture=primary_gesture,
                intent_name=spec["intent"],
                action=spec.get("action"),
                label=spec["label"],
            )
            loaded += 1

    logger.info("Loaded %d combo rule(s) from config.yaml.", loaded)
    return loaded


def classify_combo(second_hand_present: bool, primary_gesture: str, context: str) -> Intent | None:
    """
    Resolve a two-hand combo, or return None if no combo rule applies —
    callers should fall through to the normal single-hand classify() path
    when this returns None, rather than treating it as a no-op intent.

    second_hand_present just means "a second hand is visible", regardless
    of what pose it's holding — see the COMBO_RULES comment for why.
    """
    if not second_hand_present:
        return None

    context_combos = COMBO_RULES.get(context, {})
    default_combos = COMBO_RULES.get(DEFAULT_CONTEXT, {})

    entry = context_combos.get(primary_gesture) or default_combos.get(primary_gesture)
    if not entry:
        return None

    name, action, label = entry
    return Intent(name=name, action=action, label=label, context=context)
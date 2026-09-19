"""
intent_classifier.py
Maps (gesture, app-context) pairs to Intent objects via a rule table.

Every context-dependent decision lives here rather than scattered across
the main loop. Rules are overridable from config.yaml at startup.
"""

from __future__ import annotations

from events import Intent, GestureEvent

DEFAULT_CONTEXT = "default"

# context -> gesture -> (intent_name, action_or_None, label)
RULES: dict[str, dict[str, tuple[str, str | None, str]]] = {
    "default": {
        "two_finger_tap":           ("right_click",    "right_click",      "RIGHT_CLICK"),
        "thumbs_up":                ("volume_up",      "volume_up",        "VOLUME_UP"),
        "thumbs_down":              ("volume_down",    "volume_down",      "VOLUME_DOWN"),
        "closed_fist":              ("pause_toggle",   None,               "PAUSE_TOGGLE"),
        "three_fingers_up":         ("scroll",         None,               "SCROLL"),
        "open_palm_horizontal":     ("none",           None,               "-"),
        "primary_click_swipe":      ("none",           None,               "-"),
        "two_hand_swipe":           ("window_switch",  None,               "WINDOW_SWITCH"),
        "two_hand_swipe_vertical":  ("fast_scroll",    None,               "FAST_SCROLL"),
        "two_hand_pointing_up":     ("new_window",     "key:win+n",        "NEW_WINDOW (2H)"),
        "two_hand_closed_fist":     ("show_desktop",   "show_desktop",     "SHOW_DESKTOP (2H)"),
    },
    "browser": {
        "primary_click_swipe":      ("browser_nav",        None,           "BACK/FORWARD"),
        "two_hand_swipe":           ("browser_tab_switch", None,           "TAB_SWITCH"),
        "two_hand_swipe_vertical":  ("zoom_inout",         None,           "ZOOM (2H)"),
        "two_hand_pointing_up":     ("new_tab",            "key:ctrl+t",   "NEW_TAB (2H)"),
        "two_hand_closed_fist":     ("close_tab",          "key:ctrl+w",   "CLOSE_TAB (2H)"),
    },
    "video_player": {
        "three_fingers_up":         ("volume_step",     None,               "VOLUME_STEP"),
        "open_palm_horizontal":     ("video_seek",      None,               "SEEK"),
        "two_hand_swipe_vertical":  ("media_seek",      None,               "MEDIA_SEEK (2H)"),
        "two_hand_pointing_up":     ("media_play_pause","media_play_pause", "PLAY/PAUSE (2H)"),
        "two_hand_closed_fist":     ("media_stop",      "media_stop",       "STOP (2H)"),
    },
    "code_editor": {
        "two_finger_tap":           ("right_click",     "right_click",  "CONTEXT_MENU"),
        "two_hand_swipe":           ("code_tab_switch", None,           "EDITOR_TAB (2H)"),
        "two_hand_swipe_vertical":  ("zoom_inout",      None,           "ZOOM (2H)"),
        "two_hand_pointing_up":     ("new_file",        "key:ctrl+n",   "NEW_FILE (2H)"),
        "two_hand_closed_fist":     ("close_tab",       "key:ctrl+w",   "CLOSE_EDITOR (2H)"),
    },
    "document_editor": {
        "open_palm_horizontal":     ("page_nav",        None,           "PAGE_NAV"),
        "two_hand_swipe_vertical":  ("zoom_inout",      None,           "ZOOM (2H)"),
        "two_hand_pointing_up":     ("new_document",    "key:ctrl+n",   "NEW_DOC (2H)"),
    },
    "terminal": {
        "two_finger_tap":           ("right_click",     "right_click",       "PASTE"),
        "two_hand_swipe":           ("history_nav",     None,                "TERMINAL_HISTORY (2H)"),
        "two_hand_pointing_up":     ("new_terminal",    "key:ctrl+shift+t",  "NEW_TERMINAL (2H)"),
    },
}


def classify(gesture: str, context: str, event: GestureEvent | None = None) -> Intent:
    """Resolve a (gesture, context) pair into an Intent."""
    context_rules = RULES.get(context, {})
    default_rules = RULES[DEFAULT_CONTEXT]
    name, action, label = context_rules.get(
        gesture, default_rules.get(gesture, ("none", None, "-"))
    )
    return Intent(name=name, action=action, label=label, context=context, source=event)


def register_rule(context: str, gesture: str, intent_name: str, action: str | None, label: str) -> None:
    """Add or update a rule at runtime."""
    RULES.setdefault(context, {})[gesture] = (intent_name, action, label)


def load_rules_from_config(config: dict) -> int:
    """Merge config.yaml context_rules into RULES. Returns count loaded."""
    import logging
    log = logging.getLogger(__name__)
    context_rules = config.get("context_rules")
    if not context_rules:
        return 0
    loaded = 0
    for context, gestures in context_rules.items():
        if not isinstance(gestures, dict):
            log.warning("context_rules.%s is not a mapping — skipped.", context)
            continue
        for gesture, spec in gestures.items():
            if not isinstance(spec, dict) or "intent" not in spec or "label" not in spec:
                log.warning("context_rules.%s.%s missing intent/label — skipped.", context, gesture)
                continue
            register_rule(context, gesture, spec["intent"], spec.get("action"), spec["label"])
            loaded += 1
    log.info("Loaded %d context rule(s) from config.", loaded)
    return loaded


# context -> primary_gesture -> (intent_name, action, label)
COMBO_RULES: dict[str, dict[str, tuple[str, str | None, str]]] = {}


def register_combo_rule(context: str, primary_gesture: str, intent_name: str, action: str | None, label: str) -> None:
    COMBO_RULES.setdefault(context, {})[primary_gesture] = (intent_name, action, label)


def load_combo_rules_from_config(config: dict) -> int:
    """Merge config.yaml combo_rules into COMBO_RULES. Returns count loaded."""
    import logging
    log = logging.getLogger(__name__)
    combo_rules = config.get("combo_rules")
    if not combo_rules:
        return 0
    loaded = 0
    for context, gestures in combo_rules.items():
        if not isinstance(gestures, dict):
            log.warning("combo_rules.%s is not a mapping — skipped.", context)
            continue
        for primary_gesture, spec in gestures.items():
            if not isinstance(spec, dict) or "intent" not in spec or "label" not in spec:
                log.warning("combo_rules.%s.%s missing intent/label — skipped.", context, primary_gesture)
                continue
            register_combo_rule(context, primary_gesture, spec["intent"], spec.get("action"), spec["label"])
            loaded += 1
    log.info("Loaded %d combo rule(s) from config.", loaded)
    return loaded


def classify_combo(second_hand_present: bool, primary_gesture: str, context: str) -> Intent | None:
    """Return a combo Intent if a rule applies, otherwise None."""
    if not second_hand_present:
        return None
    entry = (COMBO_RULES.get(context, {}).get(primary_gesture)
             or COMBO_RULES.get(DEFAULT_CONTEXT, {}).get(primary_gesture))
    if not entry:
        return None
    name, action, label = entry
    return Intent(name=name, action=action, label=label, context=context)

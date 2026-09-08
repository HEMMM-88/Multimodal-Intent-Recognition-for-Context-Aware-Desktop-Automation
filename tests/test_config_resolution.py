"""
tests/test_config_resolution.py
Unit tests for config loading, intent classification, and config validation.

These tests run without a camera, MediaPipe, or pyautogui.
Run with:  python -m pytest tests/ -v
       or: python -m unittest tests/test_config_resolution.py -v
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from intent_classifier import (
    classify,
    classify_combo,
    load_rules_from_config,
    load_combo_rules_from_config,
    RULES,
    COMBO_RULES,
)
from events import Intent
from config_validator import validate_config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_project_config() -> dict:
    """Load the real config.yaml from the project root."""
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Intent classifier tests ───────────────────────────────────────────────────

class TestClassifyDefaults(unittest.TestCase):
    """classify() with built-in RULES (no config loaded)."""

    def test_right_click_default(self):
        intent = classify("two_finger_tap", "default")
        self.assertEqual(intent.name, "right_click")
        self.assertEqual(intent.action, "right_click")

    def test_volume_up_default(self):
        intent = classify("thumbs_up", "default")
        self.assertEqual(intent.name, "volume_up")

    def test_volume_down_default(self):
        intent = classify("thumbs_down", "default")
        self.assertEqual(intent.name, "volume_down")

    def test_scroll_default(self):
        intent = classify("three_fingers_up", "default")
        self.assertEqual(intent.name, "scroll")
        self.assertIsNone(intent.action)   # stateful — no fire-once action

    def test_pause_toggle_default(self):
        intent = classify("closed_fist", "default")
        self.assertEqual(intent.name, "pause_toggle")

    def test_window_switch_default(self):
        intent = classify("two_hand_swipe", "default")
        self.assertEqual(intent.name, "window_switch")

    def test_unknown_gesture_returns_none(self):
        intent = classify("nonexistent_gesture", "default")
        self.assertEqual(intent.name, "none")
        self.assertIsNone(intent.action)

    def test_unknown_context_falls_back_to_default(self):
        # An unknown context should fall back to the default rule.
        intent_unknown = classify("thumbs_up", "totally_unknown_context")
        intent_default = classify("thumbs_up", "default")
        self.assertEqual(intent_unknown.name, intent_default.name)


class TestClassifyContextOverrides(unittest.TestCase):
    """Context-specific rules override the default correctly."""

    def test_browser_swipe_overrides_default(self):
        intent = classify("two_hand_swipe", "browser")
        self.assertEqual(intent.name, "browser_tab_switch")

    def test_video_scroll_overrides_default(self):
        # In video_player, three_fingers_up → volume_step, not scroll
        intent = classify("three_fingers_up", "video_player")
        self.assertEqual(intent.name, "volume_step")

    def test_video_seek_horizontal(self):
        intent = classify("open_palm_horizontal", "video_player")
        self.assertEqual(intent.name, "video_seek")

    def test_code_editor_tab_switch(self):
        intent = classify("two_hand_swipe", "code_editor")
        self.assertEqual(intent.name, "code_tab_switch")

    def test_document_page_nav(self):
        intent = classify("open_palm_horizontal", "document_editor")
        self.assertEqual(intent.name, "page_nav")

    def test_terminal_right_click_is_paste(self):
        # Same action (right_click), different label — shows context awareness
        intent = classify("two_finger_tap", "terminal")
        self.assertEqual(intent.action, "right_click")
        self.assertEqual(intent.label, "PASTE")

    def test_intent_carries_context(self):
        intent = classify("thumbs_up", "browser")
        self.assertEqual(intent.context, "browser")

    def test_all_known_contexts_resolve(self):
        contexts = ["default", "browser", "video_player", "code_editor", "document_editor", "terminal"]
        for ctx in contexts:
            intent = classify("two_finger_tap", ctx)
            self.assertIsInstance(intent, Intent, f"classify() failed for context '{ctx}'")


class TestLoadRulesFromConfig(unittest.TestCase):
    """load_rules_from_config() correctly merges YAML overrides."""

    def test_loads_project_config_without_error(self):
        config = _load_project_config()
        count = load_rules_from_config(config)
        self.assertGreater(count, 0, "Expected at least one rule loaded from config.yaml")

    def test_malformed_context_is_skipped(self):
        bad_config = {
            "context_rules": {
                "default": "not_a_dict",   # should be skipped, not crash
            }
        }
        count = load_rules_from_config(bad_config)
        self.assertEqual(count, 0)

    def test_missing_intent_key_is_skipped(self):
        bad_config = {
            "context_rules": {
                "default": {
                    "thumbs_up": {"label": "UP", "action": "volume_up"}  # no 'intent'
                }
            }
        }
        count = load_rules_from_config(bad_config)
        self.assertEqual(count, 0)

    def test_missing_label_key_is_skipped(self):
        bad_config = {
            "context_rules": {
                "default": {
                    "thumbs_up": {"intent": "volume_up", "action": "volume_up"}  # no 'label'
                }
            }
        }
        count = load_rules_from_config(bad_config)
        self.assertEqual(count, 0)

    def test_valid_override_applied(self):
        override_config = {
            "context_rules": {
                "default": {
                    "thumbs_up": {
                        "intent": "custom_intent",
                        "action": "key:f1",
                        "label": "CUSTOM",
                    }
                }
            }
        }
        load_rules_from_config(override_config)
        intent = classify("thumbs_up", "default")
        self.assertEqual(intent.name, "custom_intent")
        self.assertEqual(intent.action, "key:f1")
        # Restore default so other tests aren't affected
        RULES["default"]["thumbs_up"] = ("volume_up", "volume_up", "VOLUME_UP")


class TestComboRules(unittest.TestCase):
    """classify_combo() and load_combo_rules_from_config()."""

    def test_no_combo_when_second_hand_absent(self):
        result = classify_combo(second_hand_present=False, primary_gesture="thumbs_up", context="default")
        self.assertIsNone(result)

    def test_combo_fires_when_second_hand_present(self):
        # Load the real config so COMBO_RULES is populated.
        config = _load_project_config()
        load_combo_rules_from_config(config)
        result = classify_combo(second_hand_present=True, primary_gesture="thumbs_up", context="default")
        # After loading config.yaml, thumbs_up combo → fullscreen
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "fullscreen_toggle")

    def test_combo_context_fallback(self):
        config = _load_project_config()
        load_combo_rules_from_config(config)
        # Unknown context should fall back to default combo rules
        result = classify_combo(second_hand_present=True, primary_gesture="thumbs_up", context="unknown_ctx")
        self.assertIsNotNone(result)

    def test_non_combo_gesture_returns_none(self):
        result = classify_combo(second_hand_present=True, primary_gesture="open_palm", context="default")
        self.assertIsNone(result)

    def test_malformed_combo_config_is_skipped(self):
        bad = {"combo_rules": {"default": "not_a_dict"}}
        count = load_combo_rules_from_config(bad)
        self.assertEqual(count, 0)


# ── Config validation tests ───────────────────────────────────────────────────

class TestConfigValidator(unittest.TestCase):
    """config_validator.validate_config() catches errors and warnings."""

    def test_valid_project_config_passes(self):
        config = _load_project_config()
        warnings, errors = validate_config(config)
        self.assertEqual(errors, [], f"Project config.yaml has validation errors: {errors}")

    def test_out_of_range_detection_confidence(self):
        config = _load_project_config()
        config["settings"]["detection_confidence"] = 1.5  # > 1.0
        _, errors = validate_config(config)
        self.assertTrue(any("detection_confidence" in e for e in errors))

    def test_negative_debounce_flagged(self):
        config = _load_project_config()
        config["settings"]["control"]["debounce_seconds"] = -0.1
        _, errors = validate_config(config)
        self.assertTrue(any("debounce_seconds" in e for e in errors))

    def test_invalid_primary_click_gesture(self):
        config = _load_project_config()
        config["settings"]["control"]["primary_click_gesture"] = "wave"
        _, errors = validate_config(config)
        self.assertTrue(any("primary_click_gesture" in e for e in errors))

    def test_unknown_context_in_rules_is_warning(self):
        config = _load_project_config()
        config["context_rules"] = {"browsers": {"two_finger_tap": {"intent": "x", "label": "y"}}}
        warnings, errors = validate_config(config)
        self.assertTrue(any("browsers" in w for w in warnings))
        # Should be a warning, not an error
        self.assertFalse(any("browsers" in e for e in errors))

    def test_missing_intent_in_context_rules_is_error(self):
        config = _load_project_config()
        config["context_rules"] = {"default": {"two_finger_tap": {"label": "CLICK"}}}  # no 'intent'
        _, errors = validate_config(config)
        self.assertTrue(any("intent" in e for e in errors))

    def test_bad_type_for_boolean_key_is_warning(self):
        config = _load_project_config()
        config["settings"]["show_overlay"] = "yes"  # string, not bool
        warnings, _ = validate_config(config)
        self.assertTrue(any("show_overlay" in w for w in warnings))


# ── app_detector tests ────────────────────────────────────────────────────────

class TestMatchAppConfig(unittest.TestCase):
    """match_app_config() correctly matches window titles to app keys."""

    def setUp(self):
        from app_detector import match_app_config
        self.match = match_app_config
        config = _load_project_config()
        self.apps = config.get("apps", {})

    def test_chrome_matches_browser(self):
        result = self.match("Google Chrome - New Tab", "chrome.exe", self.apps)
        self.assertEqual(result, "browser")

    def test_vscode_matches_code_editor(self):
        result = self.match("main.py - Visual Studio Code", "Code.exe", self.apps)
        self.assertEqual(result, "code_editor")

    def test_vlc_matches_video_player(self):
        result = self.match("VLC media player", "vlc.exe", self.apps)
        self.assertEqual(result, "video_player")

    def test_unknown_window_returns_none(self):
        result = self.match("Some Random App v2.0", "random.exe", self.apps)
        self.assertIsNone(result)

    def test_case_insensitive_match(self):
        result = self.match("GOOGLE CHROME", "CHROME.EXE", self.apps)
        self.assertEqual(result, "browser")


if __name__ == "__main__":
    unittest.main(verbosity=2)

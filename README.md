# MIRCADA — Multimodal Intent Recognition for Context-Aware Desktop Automation

A real-time, camera-based hand gesture system that controls your desktop using
a standard webcam. Gestures are interpreted differently depending on which
application is active — the same motion means different things in a browser,
a video player, a code editor, and so on.

---

## Architecture

```
Webcam frames
    │
    ▼
gesture_detector.py   — MediaPipe landmarks → gesture label + confidence
    │
    ▼
intent_classifier.py  — (gesture, app-context) → Intent
    │                   Rule table, YAML-driven, context-aware fallback
    ▼
action_executor.py    — Intent.action → OS action (click, key, volume, …)

app_detector.py       — Win32 foreground window → app context string
events.py             — Shared dataclasses (GestureEvent, Intent, ContextState)
config_validator.py   — Startup YAML validation with typed range checks
gesture_logger.py     — Per-event CSV logging for session analysis
main.py               — Capture loop, two-hand state machines, HUD overlay
```

---

## Core Gestures

| Gesture | Shape | Default action |
|---|---|---|
| `open_palm` 🖐 | All 4 fingers extended | Move cursor (EMA-smoothed) |
| `pointing_up` ☝ | Index finger only | Left click / drag |
| `two_finger_tap` ✌ | Index + middle up | Right click |
| `three_fingers_up` 🤟 | Index + middle + ring up | Scroll mode |
| `thumbs_up` 👍 | Thumb up, others closed | Volume +5% |
| `thumbs_down` 👎 | Thumb down, others closed | Volume −5% |
| `closed_fist` ✊ | All fingers down | Pause / Resume toggle |

Run `python main.py --list-gestures` to print this table with live config overrides.

---

## Context-Aware Behaviour

The same gesture does different things depending on the active window:

| Context | Gesture | Action |
|---|---|---|
| **browser** | `two_hand_swipe` (both open_palm, sideways) | Switch tabs |
| **browser** | `pointing_up` swipe (pinch mode) | Back / Forward |
| **video_player** | `open_palm` horizontal | Seek forward / back |
| **video_player** | `three_fingers_up` | Volume step |
| **code_editor** | `two_hand_swipe` | Switch editor tabs |
| **document_editor** | `open_palm` horizontal | Page up / down |
| **terminal** | `two_finger_tap` | Paste (right-click) |

All mappings live in `config.yaml` — no code changes needed.

---

## Two-Hand Gestures

Raise your non-primary (modifier) hand to unlock additional actions:

| Gesture combo | Action |
|---|---|
| Both hands `open_palm` → swipe left/right | Switch windows / tabs (context-aware) |
| Both hands `open_palm` → swipe up/down | Fast scroll / zoom / slide nav |
| Both hands `pointing_up` | New window / tab (context-aware) |
| Both hands `closed_fist` | Show desktop / close tab |
| Modifier `three_fingers_up` + primary `open_palm` | Screenshot |
| Modifier hand present + primary `thumbs_up` | Fullscreen toggle |
| Modifier hand present + primary `thumbs_down` | Minimize window |

The HUD overlay shows **COMBO READY** in orange when a combo is primed.

---

## Requirements

- Windows 10 / 11
- Python 3.10+
- Webcam (720p or higher recommended for long-range detection)

---

## Quick Start

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell
# or
activate_venv.cmd                  # Command Prompt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |

---

## Command-Line Options

```
python main.py                         Normal run
python main.py --config custom.yaml    Use a different config file
python main.py --no-overlay            Headless (no camera window, better focus for shortcuts)
python main.py --list-gestures         Print gesture → action table and exit
python main.py --skip-validation       Skip config.yaml validation at startup
```

---

## Configuration (`config.yaml`)

Every setting is documented inline. Key sections:

```yaml
settings:
  camera_index: 0              # webcam index
  detection_confidence: 0.30   # lower = detects hands further away
  capture_width: 1280          # request higher res for long-range detection
  detection_upscale: 1.5       # enlarge frame before MediaPipe (helps cheap cams)
  logging:
    gesture_csv_enabled: true  # write per-event CSV to logs/ for analysis

  control:
    primary_hand: "Right"
    two_hand_enabled: true
    modifier_entry_seconds: 0.20    # hold time before modifier hand activates
    two_hand_settle_frames: 4       # frames both hands must be visible before swipe detects
    gesture_confidence_threshold: 0.38

  mouse_control:
    ema_alpha: 0.45            # cursor smoothing (higher = more responsive)
    deadzone_px: 1.0

context_rules:                 # per-context gesture → intent overrides
  browser:
    two_hand_swipe:
      intent: browser_tab_switch
      label: "TAB_SWITCH"

combo_rules:                   # two-hand combo overrides
  default:
    thumbs_up:
      intent: fullscreen_toggle
      action: "key:f11"
      label: "FULLSCREEN (2H)"
```

---

## Session Analysis

When `gesture_csv_enabled: true`, every gesture event is logged to
`logs/gesture_log_<timestamp>.csv` with columns:

```
timestamp_iso, elapsed_s, gesture, confidence, palm_span, size_conf,
context, app_name, intent, action, label, modifier_active, combo, fps
```

At the end of each session, a summary table is printed to the console:

```
── Gesture Session Summary ──────────────────────────
Gesture                Events  Actions fired  Action rate
----------------------------------------------------------
closed_fist                 3              3       100.0%
open_palm                 412              0         0.0%
pointing_up                87             62        71.3%
thumbs_up                  14             14       100.0%
two_finger_tap             23             21        91.3%
```

---

## Startup Installation

```bash
python install_startup.py install    # Add to Windows startup
python install_startup.py status     # Check if installed
python install_startup.py remove     # Remove from startup
```

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

40 tests covering:
- Intent classifier defaults and all context overrides
- `load_rules_from_config()` with valid and malformed inputs
- Combo rule resolution
- Config schema validation (range checks, type checks, unknown context warnings)
- `match_app_config()` window title matching

---

## Project Structure

```
├── main.py                  Capture loop, state machines, HUD overlay
├── gesture_detector.py      MediaPipe landmarks → gesture label + confidence
├── intent_classifier.py     (gesture, context) → Intent rule table
├── action_executor.py       Intent.action → OS action
├── app_detector.py          Foreground window → app context
├── events.py                Shared dataclasses
├── config_validator.py      Startup YAML validation
├── gesture_logger.py        Per-event CSV logging + session summary
├── install_startup.py       Windows startup registration
├── config.yaml              All settings and mappings
├── requirements.txt         Python dependencies
├── logs/                    Gesture event CSVs (created on first run)
├── screenshots/             Screenshots (created on first use)
└── tests/
    ├── test_config_resolution.py   Intent + validation + app-detector tests
    └── test_module_names.py        Module existence smoke tests
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera not opening | Change `camera_index` in config (try 1, 2, …) |
| Hands not detected at distance | Lower `detection_confidence` to 0.25, raise `capture_width` to 1280 |
| Gestures triggering too fast | Increase `debounce_seconds` |
| Accidental two-hand swipes | Increase `modifier_entry_seconds` or `two_hand_settle_frames` |
| Wrong hand treated as primary | Set `primary_hand: "Left"` or check `swap_handedness` |
| App not recognised | Add window title substring to `apps:` in config |
| `win32gui` import error | `pip install pywin32` then `python Scripts\pywin32_postinstall.py -install` |
| Volume control not working | Run as administrator, or install `pycaw` |

---

## Contributors

- AbhishekRathod1
- HEMANTH REDDY (HEMMM-88)

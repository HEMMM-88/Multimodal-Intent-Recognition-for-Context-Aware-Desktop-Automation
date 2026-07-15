# 🖐 Gesture Control System

A Python application that uses your **webcam + MediaPipe** to detect hand gestures and map them to system actions — per active application.

## ✨ Features

- **13 gestures** detected in real-time via MediaPipe
- **Per-app mappings**: Chrome, Firefox, PowerPoint, Spotify, VLC, VS Code, Zoom — each gets its own gesture config
- **YAML config file** — fully customizable, no coding needed
- **Swipe detection** — directional hand movement triggers nav/scroll actions
- **Windows startup** — installs as a login startup app with one command
- **Live overlay** — optional camera window showing gesture + action in real time

---

## 📋 Requirements

- Windows 10/11
- Python 3.10+
- Webcam

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

A camera window will open. Hold your hand up and gesture away!

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause/Resume |

---

## 🤌 Gesture Reference

| Gesture | Hand Shape |
|---------|-----------|
| `peace_sign` ✌ | Index + Middle fingers up |
| `thumbs_up` 👍 | Thumb up, others closed |
| `thumbs_down` 👎 | Thumb pointing down |
| `open_hand` 🖐 | All 5 fingers extended |
| `closed_fist` ✊ | All fingers closed |
| `pointing_up` ☝ | Only index finger up |
| `pinch` 🤏 | Thumb + Index close together |
| `rock_sign` 🤘 | Index + Pinky up (devil horns) |
| `ok_sign` 👌 | Thumb + Index form circle |
| `swipe_left` 👈 | Hand moving left |
| `swipe_right` 👉 | Hand moving right |
| `swipe_up` 👆 | Hand moving up |
| `swipe_down` 👇 | Hand moving down |

Run `python main.py --list-gestures` to print this table.

---

## 🎛 Default Gesture → Action Mappings

### Chrome / Firefox
| Gesture | Action |
|---------|--------|
| ✌ peace_sign | Scroll up |
| 👎 thumbs_down | Scroll down |
| 👈 swipe_left | Browser Back |
| 👉 swipe_right | Browser Forward |
| 🖐 open_hand | New tab (Ctrl+T) |
| ✊ closed_fist | Close tab (Ctrl+W) |

### PowerPoint
| Gesture | Action |
|---------|--------|
| ✌ peace_sign | Next slide |
| 👎 thumbs_down | Previous slide |
| 👈 swipe_left | Previous slide |
| 👉 swipe_right | Next slide |
| 🖐 open_hand | Start slideshow (F5) |
| ✊ closed_fist | End slideshow (Escape) |

### Spotify
| Gesture | Action |
|---------|--------|
| 👍 thumbs_up | Volume up |
| 👎 thumbs_down | Volume down |
| 🖐 open_hand | Play / Pause |
| ✌ peace_sign | Next track |
| 🤘 rock_sign | Previous track |
| ✊ closed_fist | Mute |

---

## ⚙️ Configuration

Edit `config.yaml` to customize everything.

### Settings

```yaml
settings:
  cooldown_seconds: 0.8       # Delay between gesture triggers
  camera_index: 0             # Webcam number (0 = default)
  detection_confidence: 0.75  # MediaPipe sensitivity
  swipe_threshold: 80         # Pixels to register a swipe
  show_overlay: true          # Show camera window
  enabled_gestures: []        # Optional allow-list
  disabled_gestures: []       # Optional block-list
```

### Controlling gestures and actions

You can now control mappings in three ways:

1. Global allow/deny with `settings.enabled_gestures` and `settings.disabled_gestures`
2. App-level allow/deny with `apps.<app>.enabled_gestures` and `apps.<app>.disabled_gestures`
3. Per-gesture enable toggle using object mapping:

```yaml
default:
  rock_sign:
    action: media_prev
    enabled: false
```

### Adding a custom app

```yaml
apps:
  my_app:
    window_titles:
      - "My Application"     # Substring of the window title
    disabled_gestures:
      - closed_fist
    gestures:
      open_hand: key:ctrl+s   # Save
      thumbs_up: key:f5       # Refresh/run
      peace_sign: scroll_up
      closed_fist:
        action: key:ctrl+w
        enabled: false
```

### Available actions

| Action | Description |
|--------|-------------|
| `scroll_up` / `scroll_down` | Mouse wheel scroll |
| `click` / `right_click` | Mouse click |
| `media_play_pause` | Play/Pause |
| `media_next` / `media_prev` | Next/Prev track |
| `volume_up` / `volume_down` | Volume ±5% |
| `mute` | Toggle mute |
| `slide_next` / `slide_prev` | Arrow keys |
| `screenshot` | Save screenshot to folder |
| `show_desktop` | Win+D |
| `key:ctrl+z` | Any key combination |
| `nothing` | No action |

---

## 🖥 Startup Installation

To launch Gesture Control automatically when Windows starts:

```bash
# Install
python install_startup.py install

# Check status
python install_startup.py status

# Remove
python install_startup.py remove
```

This places a `.bat` launcher in your Windows Startup folder (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`).

---

## 🔧 Command-Line Options

```bash
python main.py                        # Normal run
python main.py --config custom.yaml   # Use custom config file
python main.py --no-overlay           # Run headless (no camera window)
python main.py --list-gestures        # Print gesture reference
```

---

## 📁 Project Structure

```
gesture_control/
├── main.py               # Main loop + overlay rendering
├── gesture_detector.py   # MediaPipe landmark → gesture name
├── action_executor.py    # Gesture action → system action
├── app_detector.py       # Active window detection
├── install_startup.py    # Windows startup registration
├── config.yaml           # All customizable settings + mappings
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not opening | Change `camera_index` in config (try 1, 2) |
| Gestures triggering too fast | Increase `cooldown_seconds` |
| Swipes not detected | Decrease `swipe_threshold` |
| App not recognized | Add window title substring to `apps:` config |
| `win32gui` import error | Run `pip install pywin32` then `python Scripts/pywin32_postinstall.py -install` |
| Volume control not working | Run as administrator or install `pycaw` |

---

## 👥 Contributors

- AbhishekRathod1
- HEMANTH REDDY (HEMMM-88)

---

## 📝 Logs

All gesture events are logged to `gesture_control.log` in the project folder for debugging.

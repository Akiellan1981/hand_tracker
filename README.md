# Hand Tracker

Real-time hand tracking from a webcam: detects both hands, tells left from
right, reads finger positions, and turns hand shapes into actions. Two
things live on top of that:

1. **Custom actions** — record any pose or short motion yourself, name it,
   and have it log/print/run a command whenever it's recognized again.
2. **Air-cursor control** — a fixed gesture vocabulary that drives your
   OS mouse cursor: point to move, pinch to zoom, grab to drag, and more.

Built on [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(21-point landmarks + left/right classification) and OpenCV, with
[PyAutoGUI](https://pyautogui.readthedocs.io/) driving the OS cursor.

## Setup

Requires a webcam and a display (this is a desktop app, not headless).

```bash
cd hand-tracker
python3 -m pip install -r requirements.txt
python3 main.py
```

The hand landmark model (~7.8MB) downloads automatically on first run into
`models/hand_landmarker.task`.

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `m` | Toggle air-cursor control on/off |
| `r` | Record a new custom action (prompts appear in the terminal) |
| `l` | List saved custom actions (printed to the terminal) |
| `d` | Delete a saved custom action |
| `h` | Toggle the on-screen help text |

## Recording your own actions

Press `r`, then answer the terminal prompts:

- **Name** — whatever you want to call it.
- **Hand** — `Left`, `Right`, or `Any`.
- **Pose or Motion** — `pose` for a held shape (e.g. a thumbs-up),
  `motion` for a short movement (e.g. a wave or swipe).
- **Threshold** — how strict the match is; blank uses the config default.
  Lower = stricter match, higher = more forgiving.
- **Command** — an optional shell command to run every time this action
  fires (e.g. `notify-send 'Hi!'`, a script, a keystroke tool). Leave
  blank to just log it.

After a 2-second countdown, hold the pose (or perform the motion) in
front of the camera. Every recognized action is printed to the terminal
and appended to `data/action_log.csv` with a timestamp.

Actions themselves are stored in `data/actions.json`; every time one
fires it's appended to `data/action_log.csv`. See
`data/actions.example.json` for the on-disk shape (the `template` values
there are placeholders; real ones are captured automatically when you
press `r`, not hand-written).

Recognition matches are pose distance or motion-sequence
[DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping) distance
against the recorded template, whichever is set per-action, so it's
robust to your hand being a bit closer/farther from the camera and to
performing a motion a bit faster or slower than when you recorded it.

## Air-cursor control

Toggle with `m`. While it's on, the **controlling hand** (`Right` by
default — see `config.json`) drives the OS cursor with a fixed gesture
vocabulary based on which fingers are extended. The other hand stays free
to trigger your own custom actions.

| Gesture | Fingers | Action |
|---|---|---|
| Point | Index only | Move the cursor |
| Pinch-select | Index + middle, held steady ~3s | Select and open (double-click) |
| Grab | Index + middle + ring | Click-and-drag while held |
| Scroll | Index + middle + ring + pinky | Hand moving up/down scrolls up/down |
| Zoom | Thumb + index only, others curled | Spread them apart to zoom in, bring together to zoom out |
| Fist | All fingers curled | Select All (Ctrl+A / Cmd+A) |
| Open palm | All 5 fingers extended | Calibrate: recenter the cursor and release any active select/drag |

The select-and-open dwell timer shows a small progress bar next to your
hand while it charges up; move more than a few pixels and it resets. Open
palm is your safety reset if a drag gets stuck or you want to recenter.

PyAutoGUI's fail-safe stays on: slam the real mouse into a screen corner
at any time to immediately abort cursor control.

## Customization

Everything tunable lives in `config.json` (partial edits are fine — any
key you don't set falls back to the built-in default):

- `camera_index`, `frame_width`/`frame_height`, `mirror` — capture setup.
  Lower resolution = higher FPS = lower input latency.
- `min_detection_confidence`, `min_tracking_confidence` — detection
  strictness.
- `custom_action_recognition` — thresholds, cooldown between re-triggers,
  how many frames a pose/motion recording captures.
- `cursor_control.controlling_hand` — which hand (`Left`/`Right`) drives
  the cursor.
- `cursor_control.control_region` — the part of the camera frame that
  maps to the full screen (0..1 normalized), so you don't need to reach
  the frame's edges.
- `cursor_control.smoothing_alpha` — 0..1, how much each new frame
  influences the cursor position. Higher = faster/twitchier response,
  lower = smoother but laggier.
- `cursor_control.dwell_seconds` / `dwell_stability_radius_px` — how long
  and how steady you need to hold the select gesture.
- `cursor_control.scroll_sensitivity` / `zoom_sensitivity` (and their
  `_dead_zone`/`_invert` siblings) — tune scroll/zoom speed and direction.

## Architecture

The recognition/matching logic has no camera or OS dependency, so it's
unit-tested directly with synthetic data (`tests/`, run with
`python3 -m pytest`):

- `src/landmarks.py` — finger extension detection, pose normalization,
  DTW motion matching.
- `src/gesture_modes.py` — the fixed cursor gesture vocabulary.
- `src/cursor_map.py` — screen-space mapping, EMA smoothing, the dwell
  selector.
- `src/motion_gestures.py` — turns continuous movement into discrete
  scroll/zoom ticks.
- `src/action_store.py`, `src/recognizer.py`, `src/session_recorder.py` —
  the custom-action record/store/match pipeline.

The camera/OS-facing layers are thin wrappers around these:
`src/hand_detector.py` (MediaPipe), `src/cursor_controller.py`
(PyAutoGUI), `app.py` (the camera loop and on-screen overlay).

## Known limitations

- Finger-extension detection compares each fingertip's distance from the
  wrist to its pip joint's distance — simple and rotation-tolerant, but
  it can occasionally misread a thumb tucked awkwardly across the palm.
- A single 2D camera has no true depth, so the zoom gesture's "spread
  apart / bring together" is measured as on-screen distance between
  thumb and index tip, not physical depth.
- Air-cursor control needs a real display; it's a no-op (with a one-time
  warning) if PyAutoGUI can't find one — expected in headless/CI
  environments.

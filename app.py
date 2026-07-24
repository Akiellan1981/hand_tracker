"""Live hand-tracking application: camera in, hand overlay + custom
action recognition + optional air-cursor control out.

Controls (shown on-screen too):
    q  - quit
    m  - toggle cursor control on/off
    r  - record a new custom action (prompts appear in the terminal)
    l  - list saved custom actions (printed to the terminal)
    d  - delete a saved custom action (prompts for a name)
    h  - toggle the help overlay
"""
import csv
import os
import shlex
import subprocess
import sys
import time

import cv2
import numpy as np

from src import config as config_module
from src import cursor_map
from src import gesture_modes
from src import landmarks
from src import setup_model
from src.action_store import Action, ActionStore
from src.cursor_controller import CursorController, select_all_keys
from src.hand_detector import HAND_CONNECTIONS, HandDetector
from src.recognizer import HandMotionBuffer, Recognizer
from src.session_recorder import MotionRecorder, PoseRecorder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEFT_COLOR = (255, 140, 0)   # BGR: blue-ish for Left hand
RIGHT_COLOR = (0, 200, 0)    # BGR: green for Right hand


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


class Toast:
    """A short-lived on-screen confirmation message."""

    def __init__(self):
        self._message = None
        self._expires_at = 0.0

    def show(self, message, duration=1.2):
        self._message = message
        self._expires_at = time.time() + duration

    def current(self):
        if self._message is not None and time.time() < self._expires_at:
            return self._message
        return None


class HandTrackerApp:
    def __init__(self, config_path):
        self.config = config_module.load_config(config_path)
        self._check_camera_available()

        self.model_path = resolve_path(self.config["model_path"])
        setup_model.download_if_missing(self.model_path)

        self.actions_path = resolve_path(self.config["actions_file"])
        self.log_path = resolve_path(self.config["log_file"])
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

        self.store = ActionStore(self.actions_path)

        cr = self.config["custom_action_recognition"]
        self.recognition_enabled = cr["enabled"]
        self.pose_threshold_default = cr["pose_match_threshold"]
        self.motion_threshold_default = cr["motion_match_threshold"]
        self.motion_sequence_length = cr["motion_sequence_length"]
        self.motion_buffer_frames = cr["motion_buffer_frames"]
        self.pose_recording_frames = cr["pose_recording_frames"]
        self.motion_recording_max_frames = cr["motion_recording_max_frames"]
        self.recognizer = Recognizer(
            self.store,
            motion_sequence_length=self.motion_sequence_length,
            cooldown_seconds=cr["recognition_cooldown_seconds"],
        )

        cc = self.config["cursor_control"]
        self.cursor_enabled = cc["enabled_by_default"]
        self.toggle_key = cc["toggle_key"]
        self.controlling_hand = cc["controlling_hand"]
        self.control_region = cc["control_region"]
        self.dwell_seconds = cc["dwell_seconds"]
        self.dwell_stability_radius_px = cc["dwell_stability_radius_px"]
        self.scroll_invert = cc["scroll_invert"]
        self.zoom_invert = cc["zoom_invert"]

        self.cursor = CursorController()
        self.screen_w, self.screen_h = self.cursor.screen_size()
        self.smoother = cursor_map.EMASmoother(alpha=cc["smoothing_alpha"])
        self.dwell_selector = cursor_map.DwellSelector(
            stability_radius=self.dwell_stability_radius_px,
            dwell_seconds=self.dwell_seconds,
        )
        self.mode_tracker = gesture_modes.ModeTracker()
        from src.motion_gestures import RelativeDeltaTracker
        self.scroll_tracker = RelativeDeltaTracker(
            sensitivity=cc["scroll_sensitivity"], dead_zone=cc["scroll_dead_zone"], invert=self.scroll_invert
        )
        self.zoom_tracker = RelativeDeltaTracker(
            sensitivity=cc["zoom_sensitivity"], dead_zone=cc["zoom_dead_zone"], invert=self.zoom_invert
        )
        self._dragging = False

        self.detector = HandDetector(
            model_path=self.model_path,
            max_num_hands=self.config["max_num_hands"],
            min_detection_confidence=self.config["min_detection_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"],
            min_hand_presence_confidence=self.config["min_hand_presence_confidence"],
        )

        self._motion_buffers = {}  # hand label -> HandMotionBuffer
        self.toast = Toast()
        self.show_help = True
        self._recording = None  # active PoseRecorder/MotionRecorder session, if any

    def _check_camera_available(self):
        """Probe the camera before the (slow) model download and MediaPipe
        init, so a missing/busy webcam fails immediately instead of after
        several seconds of unrelated setup."""
        probe = cv2.VideoCapture(self.config["camera_index"])
        available = probe.isOpened()
        probe.release()
        if not available:
            raise RuntimeError(
                f"Could not open camera index {self.config['camera_index']}. "
                "Check that a webcam is connected, not in use by another app, "
                "and that camera_index in config.json matches it."
            )

    # -- recording -------------------------------------------------------

    def _motion_buffer_for(self, hand_label):
        buf = self._motion_buffers.get(hand_label)
        if buf is None:
            buf = HandMotionBuffer(maxlen=self.motion_buffer_frames)
            self._motion_buffers[hand_label] = buf
        return buf

    def _prompt_new_action(self):
        print("\n--- Record a new action ---")
        name = input("Action name: ").strip()
        if not name:
            print("Cancelled: name cannot be empty.")
            return None

        overwrite = False
        if name in self.store:
            answer = input(f"'{name}' already exists. Overwrite? [y/N]: ").strip().lower()
            if answer != "y":
                print("Cancelled.")
                return None
            overwrite = True

        hand = input("Trigger hand? [Left/Right/Any] (default Any): ").strip().title() or "Any"
        if hand not in ("Left", "Right", "Any"):
            print(f"Unrecognized hand '{hand}', defaulting to Any.")
            hand = "Any"

        kind = input("Pose (held) or Motion (short movement)? [pose/motion] (default pose): ").strip().lower() or "pose"
        if kind not in ("pose", "motion"):
            print(f"Unrecognized kind '{kind}', defaulting to pose.")
            kind = "pose"

        default_threshold = self.pose_threshold_default if kind == "pose" else self.motion_threshold_default
        threshold_raw = input(f"Match threshold (blank = default {default_threshold}): ").strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else default_threshold
        except ValueError:
            print(f"Invalid number, using default {default_threshold}.")
            threshold = default_threshold

        command = input("Shell command to run when detected (blank = none, just log): ").strip() or None

        recorder = PoseRecorder(self.pose_recording_frames) if kind == "pose" else \
            MotionRecorder(self.motion_sequence_length, self.motion_recording_max_frames)

        print(f"Get ready to perform '{name}' with your {hand} hand...")
        return {
            "name": name,
            "hand": hand,
            "kind": kind,
            "threshold": threshold,
            "command": command,
            "overwrite": overwrite,
            "recorder": recorder,
            "countdown_until": time.time() + 2.0,
        }

    def _finish_recording(self):
        session = self._recording
        self._recording = None
        try:
            template = session["recorder"].build_template()
        except ValueError:
            print("No frames captured (hand not visible?); recording cancelled.")
            return

        action = Action(
            name=session["name"],
            kind=session["kind"],
            hand=session["hand"],
            template=template,
            threshold=session["threshold"],
            trigger={"log": True, "print": True, "command": session["command"]},
        )
        self.store.add(action, overwrite=session["overwrite"])
        print(f"Saved action '{action.name}' ({action.kind}, hand={action.hand}, threshold={action.threshold}).")
        self.toast.show(f"Saved '{action.name}'")

    def _list_actions(self):
        actions = self.store.list()
        if not actions:
            print("\nNo custom actions saved yet.")
            return
        print("\n--- Saved actions ---")
        for a in actions:
            cmd = a.trigger.get("command") if a.trigger else None
            print(f"  {a.name}: kind={a.kind} hand={a.hand} threshold={a.threshold} command={cmd!r}")

    def _delete_action_prompt(self):
        self._list_actions()
        name = input("\nName of action to delete (blank = cancel): ").strip()
        if not name:
            print("Cancelled.")
            return
        try:
            self.store.remove(name)
            print(f"Deleted '{name}'.")
            self.toast.show(f"Deleted '{name}'")
        except KeyError:
            print(f"No action named '{name}'.")

    # -- custom action triggers -------------------------------------------

    def _fire_action(self, action, hand_label):
        timestamp = time.time()
        message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {hand_label}: {action.name}"
        if action.trigger.get("print", True):
            print(message)
        if action.trigger.get("log", True):
            is_new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp", "hand", "action"])
                writer.writerow([timestamp, hand_label, action.name])
        command = action.trigger.get("command")
        if command:
            try:
                subprocess.Popen(shlex.split(command))
            except Exception as exc:
                print(f"Failed to run command for '{action.name}': {exc}", file=sys.stderr)
        self.toast.show(f"Action: {action.name}")

    # -- cursor control ----------------------------------------------------

    def _run_cursor_control(self, raw_points, states):
        mode = gesture_modes.classify_mode(states)
        entered, exited = self.mode_tracker.update(mode)
        now = time.time()

        if mode in (gesture_modes.MOVE, gesture_modes.SELECT, gesture_modes.DRAG):
            ix, iy, _ = raw_points[landmarks.INDEX_TIP]
            sx, sy = cursor_map.map_to_screen(ix, iy, self.control_region, self.screen_w, self.screen_h)
            smooth_x, smooth_y = self.smoother.update(sx, sy)
            self.cursor.move_to(smooth_x, smooth_y)
        elif self.smoother.last is not None:
            smooth_x, smooth_y = self.smoother.last
        else:
            smooth_x, smooth_y = (self.screen_w / 2, self.screen_h / 2)

        if entered == gesture_modes.DRAG:
            self.cursor.mouse_down()
            self._dragging = True
        if exited == gesture_modes.DRAG:
            self.cursor.mouse_up()
            self._dragging = False

        if entered == gesture_modes.FIST:
            self.cursor.hotkey(*select_all_keys())
            self.toast.show("Select All")

        if entered == gesture_modes.CALIBRATE:
            self.smoother.reset()
            if self._dragging:
                self.cursor.mouse_up()
                self._dragging = False
            self.toast.show("Calibrated")

        progress, triggered = self.dwell_selector.update(
            smooth_x, smooth_y, armed=(mode == gesture_modes.SELECT), now=now
        )
        if triggered:
            self.cursor.double_click(smooth_x, smooth_y)
            self.toast.show("Selected + Opened")

        if mode == gesture_modes.SCROLL:
            wrist_y = raw_points[landmarks.WRIST][1]
            ticks = self.scroll_tracker.update(wrist_y)
            if ticks:
                self.cursor.scroll(ticks)
        elif exited == gesture_modes.SCROLL:
            self.scroll_tracker.reset()

        if mode == gesture_modes.ZOOM:
            pinch = landmarks.pinch_distance(landmarks.normalize_landmarks(raw_points))
            ticks = self.zoom_tracker.update(pinch)
            if ticks:
                self.cursor.zoom(ticks)
        elif exited == gesture_modes.ZOOM:
            self.zoom_tracker.reset()

        return mode, progress

    # -- overlay -------------------------------------------------------------

    def _draw_hand(self, frame, hand, states, mode=None, dwell_progress=None):
        h, w = frame.shape[:2]
        color = LEFT_COLOR if hand.handedness == "Left" else RIGHT_COLOR
        points_px = [(int(x * w), int(y * h)) for (x, y, _) in hand.landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points_px[a], points_px[b], color, 2)
        for p in points_px:
            cv2.circle(frame, p, 4, color, -1)

        wrist_px = points_px[0]
        label = f"{hand.handedness} ({hand.score:.2f})"
        extended = [f for f, up in states.items() if up]
        cv2.putText(frame, label, (wrist_px[0] - 20, wrist_px[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"fingers: {','.join(extended) or 'none'}", (wrist_px[0] - 20, wrist_px[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if mode is not None:
            cv2.putText(frame, f"cursor: {mode.upper()}", (wrist_px[0] - 20, wrist_px[1] + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if mode == gesture_modes.SELECT and dwell_progress is not None:
                bar_x, bar_y, bar_w, bar_h = wrist_px[0] - 20, wrist_px[1] + 80, 100, 10
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * dwell_progress), bar_y + bar_h),
                              (0, 255, 255), -1)

    def _draw_hud(self, frame, fps):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Cursor control: {'ON' if self.cursor_enabled else 'off'} ({self.controlling_hand})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        toast = self.toast.current()
        if toast:
            cv2.putText(frame, toast, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self._recording is not None:
            progress = self._recording["recorder"].progress
            cv2.putText(frame, f"RECORDING '{self._recording['name']}': {progress * 100:.0f}%",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.show_help:
            help_lines = [
                f"q: quit   {self.toggle_key}: toggle cursor control   r: record action",
                "l: list actions   d: delete action   h: hide this help",
            ]
            for i, line in enumerate(help_lines):
                cv2.putText(frame, line, (10, h - 20 - 20 * (len(help_lines) - 1 - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # -- main loop -------------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(self.config["camera_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.config['camera_index']}")

        window_name = "Hand Tracker"
        prev_time = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed; stopping.", file=sys.stderr)
                    break

                if self.config["mirror"]:
                    frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.ascontiguousarray(frame_rgb)

                hands = self.detector.detect(frame_rgb)
                seen_labels = set()

                for hand in hands:
                    seen_labels.add(hand.handedness)
                    states = landmarks.finger_states(hand.landmarks)
                    normalized = landmarks.normalize_landmarks(hand.landmarks)
                    flat = landmarks.flatten(normalized)

                    buffer = self._motion_buffer_for(hand.handedness)
                    buffer.push(flat)

                    is_controlling_hand = self.cursor_enabled and hand.handedness == self.controlling_hand
                    mode = None
                    dwell_progress = None

                    if is_controlling_hand:
                        mode, dwell_progress = self._run_cursor_control(hand.landmarks, states)
                    elif self.recognition_enabled and self._recording is None:
                        match = self.recognizer.match(hand.handedness, flat, buffer)
                        if match is not None:
                            self._fire_action(match, hand.handedness)

                    if self._recording is not None and time.time() >= self._recording["countdown_until"]:
                        target_hand = self._recording["hand"]
                        locked_hand = self._recording.get("locked_hand")
                        if locked_hand is None and target_hand in (hand.handedness, "Any"):
                            locked_hand = hand.handedness
                            self._recording["locked_hand"] = locked_hand
                        if locked_hand == hand.handedness:
                            self._recording["recorder"].add_frame(flat)
                            if self._recording["recorder"].is_complete:
                                self._finish_recording()

                    self._draw_hand(frame, hand, states, mode, dwell_progress)

                for label in list(self._motion_buffers.keys()):
                    if label not in seen_labels:
                        self._motion_buffers[label] = HandMotionBuffer(maxlen=self.motion_buffer_frames)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                self._draw_hud(frame, fps)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord(self.toggle_key):
                    self.cursor_enabled = not self.cursor_enabled
                    self.smoother.reset()
                    if self._dragging:
                        self.cursor.mouse_up()
                        self._dragging = False
                    print(f"Cursor control {'enabled' if self.cursor_enabled else 'disabled'}.")
                elif key == ord("r"):
                    if self._recording is None:
                        session = self._prompt_new_action()
                        if session is not None:
                            self._recording = session
                    else:
                        print("Recording cancelled.")
                        self.toast.show("Recording cancelled")
                        self._recording = None
                elif key == ord("l"):
                    self._list_actions()
                elif key == ord("d"):
                    self._delete_action_prompt()
                elif key == ord("h"):
                    self.show_help = not self.show_help
        finally:
            if self._dragging:
                self.cursor.mouse_up()
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

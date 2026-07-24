"""Matches a live hand's landmarks against the user's custom recorded
actions (pose or motion) and reports the best match, debounced by a
per-action cooldown so a held pose doesn't re-fire every frame.
"""
import time
from collections import deque

from . import landmarks


class HandMotionBuffer:
    """Rolling window of recent normalized/flattened frames for one hand,
    used to match motion (sequence) actions."""

    def __init__(self, maxlen):
        self._buffer = deque(maxlen=maxlen)

    def push(self, flat_vector):
        self._buffer.append(flat_vector)

    def snapshot(self):
        return list(self._buffer)

    def __len__(self):
        return len(self._buffer)


class Recognizer:
    def __init__(self, store, motion_sequence_length, cooldown_seconds):
        self.store = store
        self.motion_sequence_length = motion_sequence_length
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered = {}

    def _on_cooldown(self, name, now):
        last = self._last_triggered.get(name)
        return last is not None and (now - last) < self.cooldown_seconds

    def match(self, hand_label, flat_pose_vector, motion_buffer, now=None):
        """Returns the best-matching Action for this hand this frame, or
        None. Checks pose actions against the current frame and motion
        actions against the hand's rolling motion buffer."""
        now = time.time() if now is None else now
        best_action = None
        best_distance = None

        for action in self.store.list():
            if action.hand != "Any" and action.hand != hand_label:
                continue
            if self._on_cooldown(action.name, now):
                continue

            if action.kind == "pose":
                distance = landmarks.euclidean_distance(flat_pose_vector, action.template)
            else:
                frames = motion_buffer.snapshot()
                if len(frames) < 2:
                    continue
                resampled = landmarks.resample_sequence(frames, self.motion_sequence_length)
                distance = landmarks.dtw_distance(resampled, action.template)

            if distance <= action.threshold and (best_distance is None or distance < best_distance):
                best_action = action
                best_distance = distance

        if best_action is not None:
            self._last_triggered[best_action.name] = now

        return best_action

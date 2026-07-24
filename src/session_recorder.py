"""Builds an action template from a sequence of captured hand frames.
Pure logic: callers feed in normalized landmark vectors (from
landmarks.normalize_landmarks + landmarks.flatten) frame by frame; no
camera/mediapipe dependency, so it's unit-testable with synthetic data.
"""
from . import landmarks


class PoseRecorder:
    """Averages a burst of frames of a held pose into a single template
    vector."""

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self._frames = []

    def add_frame(self, flat_vector):
        self._frames.append(flat_vector)

    @property
    def progress(self):
        return min(1.0, len(self._frames) / self.num_frames)

    @property
    def is_complete(self):
        return len(self._frames) >= self.num_frames

    def build_template(self):
        if not self._frames:
            raise ValueError("no frames captured")
        dim = len(self._frames[0])
        totals = [0.0] * dim
        for frame in self._frames:
            for i in range(dim):
                totals[i] += frame[i]
        n = len(self._frames)
        return [t / n for t in totals]


class MotionRecorder:
    """Captures a variable-length sequence of frames for a short motion
    gesture, then resamples it to a fixed length so it can be compared
    with dynamic time warping regardless of recording speed/duration."""

    def __init__(self, target_length, max_frames):
        self.target_length = target_length
        self.max_frames = max_frames
        self._frames = []

    def add_frame(self, flat_vector):
        if len(self._frames) < self.max_frames:
            self._frames.append(flat_vector)

    @property
    def progress(self):
        return min(1.0, len(self._frames) / self.max_frames)

    @property
    def is_complete(self):
        return len(self._frames) >= self.max_frames

    def build_template(self):
        if not self._frames:
            raise ValueError("no frames captured")
        return landmarks.resample_sequence(self._frames, self.target_length)

"""Converts continuous hand movement into discrete "ticks" (scroll wheel
clicks) for the scroll and zoom gestures. Pure logic, no OS dependency.
"""


class RelativeDeltaTracker:
    """Tracks a scalar signal (e.g. hand height, or hand size as a
    distance-from-camera proxy) frame to frame, and converts movement into
    an integer tick count once accumulated movement crosses a full tick,
    so small per-frame jitter doesn't produce noisy output."""

    def __init__(self, sensitivity, dead_zone, invert=False):
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.invert = invert
        self._last_value = None
        self._accumulator = 0.0

    def reset(self):
        self._last_value = None
        self._accumulator = 0.0

    def update(self, value):
        if self._last_value is None:
            self._last_value = value
            return 0

        delta = value - self._last_value
        self._last_value = value

        if abs(delta) < self.dead_zone:
            return 0

        if self.invert:
            delta = -delta

        self._accumulator += delta * self.sensitivity
        ticks = int(self._accumulator)
        self._accumulator -= ticks
        return ticks

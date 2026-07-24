"""Pure logic for mapping a normalized hand-landmark point to screen pixels,
smoothing it for a responsive-but-stable cursor, and detecting a "hold
still" dwell selection. No OS/camera dependency, so it is unit-testable.
"""


def map_to_screen(x_norm, y_norm, region, screen_w, screen_h):
    """Map a normalized (0..1) point through a control region (the part of
    the camera frame that counts as "the screen") to screen pixels, so the
    hand doesn't need to reach the frame's edges to reach the screen's
    edges. Values outside the region are clamped."""
    x_min, y_min = region["x_min"], region["y_min"]
    x_max, y_max = region["x_max"], region["y_max"]

    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)

    fx = (x_norm - x_min) / span_x
    fy = (y_norm - y_min) / span_y
    fx = min(1.0, max(0.0, fx))
    fy = min(1.0, max(0.0, fy))

    return fx * screen_w, fy * screen_h


class EMASmoother:
    """Exponential moving average smoother. `alpha` is the weight given to
    the newest sample: closer to 1.0 tracks the raw signal almost
    instantly (fast response, more jitter), closer to 0.0 favors smoothness
    over responsiveness."""

    def __init__(self, alpha=0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value = None

    def update(self, x, y):
        if self._value is None:
            self._value = (x, y)
        else:
            px, py = self._value
            self._value = (
                self.alpha * x + (1 - self.alpha) * px,
                self.alpha * y + (1 - self.alpha) * py,
            )
        return self._value

    def reset(self):
        self._value = None

    @property
    def last(self):
        return self._value


class DwellSelector:
    """Fires once when a point is held within `stability_radius` of its
    anchor for `dwell_seconds` while `armed` is True. Requires `armed` to
    go False and True again (the gesture released and re-entered) before
    it can fire again, so it doesn't repeat-fire every frame the pose is
    held past the threshold."""

    def __init__(self, stability_radius, dwell_seconds):
        self.stability_radius = stability_radius
        self.dwell_seconds = dwell_seconds
        self._anchor = None
        self._start_time = None
        self._latched = False

    def update(self, x, y, armed, now):
        """Returns (progress, triggered): progress is 0..1, triggered is
        True exactly once per arm-cycle when progress reaches 1.0."""
        if not armed:
            self._anchor = None
            self._start_time = None
            self._latched = False
            return 0.0, False

        if self._latched:
            return 1.0, False

        if self._anchor is None:
            self._anchor = (x, y)
            self._start_time = now
            return 0.0, False

        dx, dy = x - self._anchor[0], y - self._anchor[1]
        if (dx * dx + dy * dy) ** 0.5 > self.stability_radius:
            self._anchor = (x, y)
            self._start_time = now
            return 0.0, False

        elapsed = now - self._start_time
        progress = min(1.0, elapsed / self.dwell_seconds)
        if progress >= 1.0:
            self._latched = True
            return 1.0, True
        return progress, False
